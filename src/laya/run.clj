(ns laya.run
  "jolt -M:run <workflow> [input] [--options JSON] [--data DIR] [--workflows DIR[:DIR]]
   jolt -M:run --list

  Run one workflow (laya.workflows) against the prepared model and print the
  answer JSON. `input` is a JSON document (object, array or string) handed
  to the workflow's `state` fn, or @path to read it from a file; without it
  the state fn gets nil (demo then picks its example). --options is a JSON
  object for workflows whose `questions` take options."
  (:require [clojure.string :as str]
            [laya.agent :as ag]
            [laya.config :as cfg]
            [laya.json :as json]
            [laya.sequence :as seq]
            [laya.workflows :as wf])
  (:gen-class))

(defn- parse-json-arg [s what]
  (when s
    (let [text (if (str/starts-with? s "@") (slurp (subs s 1)) s)]
      (try (json/read-str text)
           (catch Exception e
             (throw (ex-info (str what " is not valid JSON: " (ex-message e)) {:type :invalid-request :arg what} e)))))))

(defn run-workflow
  "Load the workflows under dirs, run `name` on the JSON input (string or
  nil) with the JSON options (string or nil); answers as an ordered map."
  [agent dirs name input options]
  (let [wfs (wf/load-workflows dirs)
        w (or (get wfs name)
              (throw (ex-info (str "no workflow named " (pr-str name) "; known: "
                                   (str/join ", " (sort (keys wfs))))
                              {:type :invalid-request :workflow name :known (sort (keys wfs))})))
        state (wf/state w (parse-json-arg input "input"))
        questions (wf/questions w (or (parse-json-arg options "--options") {}))]
    (ag/system-one agent state questions)))

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
      (nil? name) (do (println (:doc (meta (find-ns 'laya.run)))) (System/exit 2))
      :else
      (try
        (let [agent (ag/load-agent (cfg/setting ctx "--data" "LAYA_DATA" :data "data"))]
          (println (seq/json-str (run-workflow agent dirs name input (get opts "--options")))))
        (catch Exception e
          (if (#{:invalid-request :invalid-question} (:type (ex-data e)))
            (do (binding [*out* *err*] (println "jolt run:" (ex-message e))) (System/exit 1))
            (throw e)))))))
