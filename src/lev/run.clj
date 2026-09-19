(ns lev.run
  "jolt -M:run <workflow> [input] [--options JSON] [--constraints JSON] [--data DIR]
              [--model NAME] [--thinking true|false] [--thinker PATH.gguf]
              [--workflows DIR[:DIR]] [--max-len N] [--head-max-len N]
   jolt -M:run decide TEXT --choices a,b,c [--instructions ...]      ; one choice
   jolt -M:run judge TEXT --instructions ...                          ; one probability
   jolt -M:run rate TEXT --levels low,mid,high [--instructions ...]   ; one score
   jolt -M:run ask @request.json                                      ; a whole systemone body
   jolt -M:run --list

  Run one workflow (lev.workflows) against a model and print the answer
  JSON. `input` is a JSON document (object, array or string) handed to the
  workflow's `state` fn, or @path to read it from a file; without it the
  state fn gets nil (demo then picks its example). --options is a JSON
  object for workflows whose `questions` take options; --constraints a JSON
  list of lev.constraints added to the workflow's own. --model is a
  checkpoint (english by default) or a thinker (config.edn :thinkers, or
  --thinker / LEV_THINKER as `thinker`), --thinking its override."
  (:require [clojure.string :as str]
            [lev.agent :as ag]
            [lev.api :as api]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.workflows :as wf])
  (:gen-class))

(defn- parse-json-arg [s what]
  (when s
    (let [text (if (str/starts-with? s "@") (slurp (subs s 1)) s)]
      (try (json/read-str text)
           (catch Exception e
             (throw (ex-info (str what " is not valid JSON: " (ex-message e)) {:type :invalid-request :arg what} e)))))))

(defn run-workflow
  "Load the workflows under dirs, run `name` on the JSON input (string or
  nil) with the JSON options (string or nil) and the JSON constraints
  (string or nil, added to the workflow's own); answers as an ordered map."
  ([agent dirs name input options] (run-workflow agent dirs name input options nil))
  ([agent dirs name input options constraints] (run-workflow agent dirs name input options constraints nil))
  ([agent dirs name input options constraints extra-opts]
   (let [wfs (wf/load-workflows dirs)
         w (or (get wfs name)
               (throw (ex-info (str "no workflow named " (pr-str name) "; known: "
                                    (str/join ", " (sort (keys wfs))))
                               {:type :invalid-request :workflow name :known (sort (keys wfs))})))
         opts (or (parse-json-arg options "--options") {})
         state (wf/state w (parse-json-arg input "input"))
         questions (wf/questions w opts)
         own (wf/constraints w opts)
         theirs (parse-json-arg constraints "--constraints")]
     (ag/system-one agent state questions
                    (merge {:constraints (when (or own theirs) (vec (concat own theirs)))}
                           extra-opts)))))

(def one-offs #{"decide" "judge" "rate" "ask"})

(defn- csv [s] (vec (remove str/blank? (map str/trim (str/split (str s) #",")))))

(defn- need [opts flag]
  (or (let [v (get opts flag)] (when (string? v) v))
      (throw (ex-info (str flag " is required") {:type :invalid-request :arg flag}))))

(defn one-off
  "The one-question conveniences as commands: decide (--choices, an
  --instructions), judge (--instructions), rate (--levels, an
  --instructions), and ask, a systemone request body (JSON or @path) run as
  is. `args` are the positionals after the command; answers the choice /
  score answer map, the probability, or the whole answer map."
  [agent cmd args opts]
  (let [text (first args)
        extra (when (contains? opts "--thinking") {:thinking (= "true" (str (get opts "--thinking")))})]
    (case cmd
      "decide" (api/decide agent text (csv (need opts "--choices"))
                           (get opts "--instructions" "Which option best describes the state?") extra)
      "judge" (api/judge agent text (need opts "--instructions") nil extra)
      "rate" (api/rate agent text (csv (need opts "--levels"))
                       (get opts "--instructions" "Rate where the state falls on this scale:") extra)
      "ask" (let [body (or (parse-json-arg text "request")
                           (throw (ex-info "ask needs a request: JSON or @path with state and questions" {:type :invalid-request :arg "request"})))]
              (when-not (and (map? body) (contains? body "state") (map? (get body "questions")))
                (throw (ex-info "the request must be an object with state and questions" {:type :invalid-request :arg "request"})))
              (ag/system-one agent (get body "state") (get body "questions")
                             (merge {:constraints (get body "constraints") :on-infeasible (get body "on_infeasible")}
                                    (when (contains? body "thinking") {:thinking (get body "thinking")})
                                    (when (contains? body "thought") {:thought (get body "thought")})
                                    extra))))))

(defn list-workflows
  "Print name, source file and description of every workflow under dirs."
  [dirs]
  (let [wfs (wf/load-workflows dirs)]
    (if (empty? wfs)
      (println "no workflows found under" (str/join ", " dirs))
      (doseq [[name w] (sort-by key wfs)]
        (println (format "%-16s %s" name (:file w)))
        (when (:doc w) (println (str (apply str (repeat 17 " ")) (first (str/split-lines (:doc w))))))))))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        dirs (cfg/workflow-dirs ctx)
        [name input] (:args opts)]
    (cond
      (get opts "--list") (list-workflows dirs)
      (nil? name) (do (println (:doc (meta (find-ns 'lev.run)))) (System/exit 2))
      :else
      (try
        (let [model (get opts "--model" "english")
              rt (router/make-router {:data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
                                      :thinkers (cfg/thinkers ctx)
                                      :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)})
              agent (router/load-model rt model)
              extra (when (contains? opts "--thinking") {:thinking (= "true" (str (get opts "--thinking")))})]
          (println (seq/json-str (if (one-offs name)
                                   (one-off agent name (rest (:args opts)) opts)
                                   (run-workflow agent dirs name input (get opts "--options") (get opts "--constraints") extra)))))
        (catch Exception e
          (if (#{:invalid-request :invalid-question :invalid-constraint :unknown-model :model-unavailable} (:type (ex-data e)))
            (do (binding [*out* *err*] (println "jolt run:" (ex-message e))) (System/exit 1))
            (throw e)))))))
