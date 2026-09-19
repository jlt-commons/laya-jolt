(ns lev.server
  "HTTP front for the checkpoints and workflows, mirroring the TypeSafe Jev
  API (https://docs.typesafe.ai/api) and adding what the Python package's
  Router and presets offer:

    POST /v1/systemone        {\"state\" ..., \"questions\" {...},
                               \"constraints\"? [...], \"on_infeasible\"?,
                               \"model\"? \"lang\"? \"task\"?}
                              -> {\"model\" \"laya-rl-agent\", \"answers\" {...},
                                  \"usage\" {...}, \"constraints\"? {...},
                                  \"routing\" {...}}
    POST /v1/route            same body, questions optional -> the routing
                              decision alone, nothing loaded or run
    POST /v1/workflows/:name  {\"input\" ..., \"options\"? {...},
                               \"constraints\"? [...], \"on_infeasible\"?,
                               \"model\"? \"lang\"? \"task\"?}
                              -> systemone answer + \"workflow\" + the built \"state\"
    GET  /v1/models           the checkpoints: repo, data dir, prepared, loaded
    GET  /v1/workflows        the loaded workflows: description, question ids
    GET  /health              {\"status\" \"ok\", \"model\" \"laya-rl-agent\",
                               \"loaded\" [...], \"workflows\" [...]}

  `model` is absent (or the engine's own name, laya-rl-agent) to route by
  content, or a checkpoint name / alias (english, multilingual,
  typed-decisions, en, ml, ...) to pick one; `lang` and `task` are the
  Router's other hints. `constraints` is a list of lev.constraints over
  the question ids (a workflow's own come first, the request's are added),
  decided jointly after the forward pass: every answer then carries
  `decided` and the body a `constraints` report; `on_infeasible` is
  min_violations (default) or raise (a 422 of type infeasible listing the
  violated constraints). Anything else is a 422. Routes are dispatched by
  ruuter.

  Auth is `Authorization: Bearer <key>` on /v1/* when the server is started
  with an :api-key (LEV_API_KEY); without one every request is accepted.
  Errors: 401 for a missing/invalid key, 422 with a FastAPI-style
  {\"detail\": [{\"loc\": [...], \"msg\": ..., \"type\": ...}]} for anything
  wrong with the body, 404 for unknown routes and workflows, 405 for the
  wrong method, 503 when the chosen checkpoint has no prepared data, 413
  from the adapter when the body exceeds :max-request-bytes.

  As a library: (handler router-or-agent opts) is a plain ring handler to
  mount in another app; (start router-or-agent opts) / (stop server) run it
  on ring-chez-adapter. As a binary: `jolt build -m lev.server -o
  lev-server` (the `binary` task), then `./lev-server --data data`.

  Requests are parsed with lev.json, which keeps object key order: the
  order of options, questions and state fields is model input. Inference
  (and checkpoint loading) runs one request at a time behind a lock; the
  adapter's workers only overlap on I/O."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.tokenizer :as tk]
            [lev.workflows :as wf]
            [ring-chez.adapter :as adapter]
            [ruuter.core :as ruuter])
  (:gen-class))

(def default-model "laya-rl-agent")

;; the engine's own names: "route for me", not a checkpoint choice
(def ^:private engine-names #{"laya-rl-agent" "rl-agent"})

;; --- responses ------------------------------------------------------------------

(defn- json-response [status body]
  {:status status
   :headers {"Content-Type" "application/json"}
   :body (str (seq/json-str body) "\n")})

(defn- detail [loc msg type]
  (seq/ordered-map [["loc" (vec loc)] ["msg" msg] ["type" type]]))

(defn- unprocessable [details]
  (json-response 422 {"detail" (vec details)}))

(def ^:private unauthorized
  (json-response 401 {"detail" "Missing or invalid API key. Check the Authorization header."}))

(def ^:private not-found (json-response 404 {"detail" "Not Found"}))
(def ^:private method-not-allowed (json-response 405 {"detail" "Method Not Allowed"}))

(defn- question-loc [{:keys [qid field]}]
  (cond-> ["body" "questions" (str qid)] field (conj field)))

(defn- error-response
  "The HTTP shape of the exceptions the library throws; rethrows the rest."
  [e]
  (let [data (ex-data e)]
    (case (:type data)
      :invalid-json (unprocessable [(detail ["body"] (ex-message e) "json_invalid")])
      :invalid-question (unprocessable [(detail (question-loc data) (ex-message e) "value_error")])
      :unknown-model (unprocessable [(detail ["body" "model"] (ex-message e) "value_error")])
      :invalid-request (unprocessable [(detail ["body" (or (:field data) "options")] (ex-message e) "value_error")])
      :invalid-constraint (unprocessable [(detail (cond-> ["body" "constraints"]
                                                    (:index data) (conj (- (:index data) (:offset data 0))))
                                                  (ex-message e) "value_error")])
      :infeasible (unprocessable [(assoc (detail ["body" "constraints"] (ex-message e) "infeasible")
                                         "violations" (:violations data))])
      :model-unavailable (json-response 503 {"detail" (ex-message e)})
      (throw e))))

;; --- request ---------------------------------------------------------------------

(defn- body-string [body]
  (cond (nil? body) "" (string? body) body :else (slurp body)))

(defn- parse-body
  "The JSON document in the request body; ex-info :invalid-json otherwise.
  With allow-empty?, an empty body reads as {}."
  ([req] (parse-body req false))
  ([req allow-empty?]
   (let [text (body-string (:body req))]
     (if (str/blank? text)
       (if allow-empty? {} (throw (ex-info "json: request body is empty" {:type :invalid-json})))
       (json/read-str text)))))

(defn- check-routing-fields
  "model / lang / task must be strings, and model a known checkpoint (or
  the engine's own name)."
  [body]
  (concat
   (for [k ["model" "lang" "task"]
         :when (and (contains? body k) (not (string? (get body k))))]
     (detail ["body" k] (str k " must be a string") "type_error"))
   (let [m (get body "model")]
     (when (and (string? m) (not (engine-names m)))
       (try (router/normalise-name m) nil
            (catch Exception e
              [(detail ["body" "model"] (ex-message e) "value_error")]))))))

(def ^:private infeasible-modes #{"min_violations" "raise"})

(defn- check-constraints
  "constraints, when present, is a list (each entry is checked against the
  questions by the library, before any inference); on_infeasible one of
  the two modes."
  [body]
  (concat
   (when (and (contains? body "constraints") (not (sequential? (get body "constraints"))))
     [(detail ["body" "constraints"] "constraints must be a list of constraints" "type_error")])
   (when (and (contains? body "on_infeasible") (not (contains? infeasible-modes (get body "on_infeasible"))))
     [(detail ["body" "on_infeasible"] "on_infeasible must be min_violations or raise" "value_error")])))

(defn- check-state [body]
  (let [state (get body "state")]
    (when-not (or (string? state) (map? state) (sequential? state))
      [(detail ["body" "state"] "state is required: a string, object or array" "value_error")])))

(defn- check-questions [qs]
  (cond
    (not (map? qs)) [(detail ["body" "questions"] "questions is required: an object of question id -> question" "value_error")]
    (empty? qs) [(detail ["body" "questions"] "questions must not be empty" "value_error")]
    :else (keep (fn [[qid qdef]]
                  (try (ag/validate-question qid qdef) nil
                       (catch Exception e
                         (detail (question-loc (ex-data e)) (ex-message e) "value_error"))))
                qs)))

(defn- check-request
  "Everything wrong with a parsed /v1/systemone body, as detail entries.
  Each question is checked with agent/validate-question, so the reasons
  match the library."
  [body]
  (if-not (map? body)
    [(detail ["body"] "request body must be a JSON object" "type_error")]
    (vec (concat (check-state body)
                 (check-routing-fields body)
                 (check-constraints body)
                 (check-questions (get body "questions"))))))

(defn- routing-opts
  "The Router hints in a body: :model (nil for the engine's own name),
  :lang, :task, plus :constraints / :on-infeasible for the decoder."
  [body]
  (let [m (get body "model")]
    {:model (when-not (engine-names m) m)
     :lang (get body "lang")
     :task (get body "task")
     :constraints (get body "constraints")
     :on-infeasible (get body "on_infeasible")}))

(defn- predict
  "router/predict under the inference lock, with the errors the decoder
  throws pointed at the request: a bad constraint at its index among the
  request's own (`offset` of them belong to the workflow)."
  [rt lock state questions {:keys [model lang task constraints on-infeasible]} offset]
  (try (locking lock
         (router/predict rt state questions :model model :lang lang :task task
                         :constraints constraints :on-infeasible on-infeasible))
       (catch Exception e
         (if (= :invalid-constraint (:type (ex-data e)))
           (throw (ex-info (ex-message e) (assoc (ex-data e) :offset offset)))
           (throw e)))))

;; --- endpoints ---------------------------------------------------------------------

(defn systemone
  "Answer one parsed /v1/systemone body: route, load if needed, infer.
  Inference is serialized on lock."
  [rt lock body]
  (let [details (check-request body)]
    (if (seq details)
      (unprocessable details)
      (json-response 200 (predict rt lock (get body "state") (get body "questions") (routing-opts body) 0)))))

(defn route-only
  "POST /v1/route: the decision for a body, without loading or running."
  [rt body]
  (let [details (if-not (map? body)
                  [(detail ["body"] "request body must be a JSON object" "type_error")]
                  (concat (check-state body)
                          (check-routing-fields body)
                          (when (and (contains? body "questions") (not (map? (get body "questions"))))
                            [(detail ["body" "questions"] "questions must be an object" "type_error")])))]
    (if (seq details)
      (unprocessable details)
      (let [{:keys [model lang task]} (routing-opts body)]
        (json-response 200 (router/route rt (get body "state") (get body "questions" {})
                                         :model model :lang lang :task task))))))

(defn run-workflow
  "POST /v1/workflows/:name: the workflow's state fn on \"input\", its
  questions with \"options\", routed and answered; the answer carries the
  workflow name and the state the model actually read."
  [rt lock workflows name body]
  (let [w (get workflows name)]
    (cond
      (nil? w)
      (json-response 404 {"detail" (str "no workflow named " (pr-str name) "; known: "
                                        (str/join ", " (sort (keys workflows))))})

      (not (map? body))
      (unprocessable [(detail ["body"] "request body must be a JSON object" "type_error")])

      (and (contains? body "options") (not (map? (get body "options"))))
      (unprocessable [(detail ["body" "options"] "options must be an object" "type_error")])

      :else
      (let [details (concat (check-routing-fields body) (check-constraints body))]
        (if (seq details)
          (unprocessable details)
          (let [state (wf/state w (get body "input"))
                questions (wf/questions w (get body "options" {}))
                details (concat (check-state {"state" state}) (check-questions questions))]
            (if (seq details)
              (unprocessable details)
              ;; the workflow's constraints first, then the request's; a
              ;; workflow that declares none leaves the answers plain
              ;; unless the request brings some
              (let [own (wf/constraints w (get body "options" {}))
                    theirs (get body "constraints")
                    opts (assoc (routing-opts body)
                                :constraints (when (or own theirs) (vec (concat own theirs))))
                    result (predict rt lock state questions opts (count own))]
                (json-response 200 (assoc result "workflow" name "state" state))))))))))

(defn- health [rt workflows]
  (json-response 200 (seq/ordered-map [["status" "ok"] ["model" default-model]
                                       ["loaded" (router/loaded rt)]
                                       ["workflows" (vec (sort (keys workflows)))]])))

(defn- models [rt]
  (let [loaded (set (router/loaded rt))]
    (json-response 200 (seq/ordered-map
                        [["default" (:default rt)]
                         ["max_loaded" (:max-loaded rt)]
                         ["models" (seq/ordered-map
                                    (for [[name dir] (:models rt)]
                                      [name (seq/ordered-map [["repo" (get router/repos name)]
                                                              ["data" dir]
                                                              ["available" (router/available? rt name)]
                                                              ["loaded" (contains? loaded name)]
                                                              ["limits" (when-let [l (router/effective-limits rt name)]
                                                                          (seq/ordered-map [["max_len" (:max-len l)]
                                                                                            ["head_max_len" (:head-max-len l)]]))]])]))]]))))

(defn- list-workflows [workflows]
  (json-response 200 {"workflows"
                      (seq/ordered-map
                       (for [[name w] (sort-by key workflows)]
                         [name (seq/ordered-map [["description" (:doc w)]
                                                 ["file" (:file w)]
                                                 ["questions" (vec (map #(if (keyword? %) (clojure.core/name %) (str %))
                                                                        (keys (wf/questions w))))]
                                                 ["constraints" (vec (wf/constraints w))]
                                                 ["options" (:options? w)]])]))}))

;; --- the handler --------------------------------------------------------------------

(defn- authorized? [req api-key]
  (or (nil? api-key)
      (= (get-in req [:headers "authorization"]) (str "Bearer " api-key))))

(def ^:private http-methods [:get :post :put :delete :patch :head :options])

(defn- routes
  "The ruuter route table. Every real route is followed by 405 entries for
  the other methods on its path; anything else is the 404."
  [rt lock workflows api-key]
  (let [guard (fn [f] (fn [req] (if (authorized? req api-key) (f req) unauthorized)))
        json-in (fn [f allow-empty?]
                  (fn [req] (try (f (parse-body req allow-empty?))
                                 (catch Exception e (error-response e)))))
        real [{:path "/health" :method :get :response (fn [_] (health rt workflows))}
              {:path "/v1/systemone" :method :post
               :response (guard (json-in #(systemone rt lock %) false))}
              {:path "/v1/route" :method :post
               :response (guard (json-in #(route-only rt %) false))}
              {:path "/v1/models" :method :get :response (guard (fn [_] (models rt)))}
              {:path "/v1/workflows" :method :get :response (guard (fn [_] (list-workflows workflows)))}
              {:path "/v1/workflows/:name" :method :post
               :response (guard (fn [req]
                                  (try (run-workflow rt lock workflows (get-in req [:params :name])
                                                     (parse-body req true))
                                       (catch Exception e (error-response e)))))}]
        disallowed (for [{:keys [path method]} real
                         m http-methods :when (not= m method)]
                     {:path path :method m :response method-not-allowed})]
    (vec (concat real disallowed [{:path :not-found :response not-found}]))))

(defn handler
  "A ring handler for the API. `rt` is a lev.router router, or one loaded
  agent (served as the english checkpoint). opts: :api-key (nil = no auth),
  :workflows {name workflow} from lev.workflows/load-workflows."
  [rt {:keys [api-key workflows]}]
  (let [rt (if (router/router? rt) rt (router/preloaded rt))
        table (routes rt (Object.) (or workflows {}) api-key)]
    (fn [req] (ruuter/route table req))))

;; --- server ----------------------------------------------------------------------

(defn start
  "Run the API on ring-chez-adapter; answers the server handle.
  opts: :port (8080), :host (\"127.0.0.1\"; \"0.0.0.0\" for all interfaces),
  :api-key, :workflows, :max-request-bytes (4 MiB), plus anything the
  adapter takes."
  [rt {:keys [port host max-request-bytes]
       :or {port 8080 host "127.0.0.1" max-request-bytes (* 4 1024 1024)}
       :as opts}]
  (adapter/run-server
   (handler rt opts)
   (merge {:on-failure (fn [_ ex]
                         (binding [*out* *err*] (println "lev.server:" (ex-message ex)))
                         (json-response 500 {"detail" "Internal Server Error"}))}
          (dissoc opts :api-key :workflows)
          {:port port :host host :max-request-bytes max-request-bytes})))

(defn stop [server]
  (adapter/stop-server server))

(defn self-test
  "Run the README quickstart through the handler and compare with the
  json.dumps(Agent.system_one(...)) pinned in <golden-dir>/readme.edn, plus
  the tokenizer paths a release build has miscompiled before. Answers
  [[name ok? detail] ...]. The test suite runs interpreted; this is what
  proves the AOT binary computes the same thing."
  [agent golden-dir]
  (let [h (handler agent {})
        golden (fn [name] (edn/read-string {:readers {'laya/omap seq/ordered-map}}
                                           (slurp (str golden-dir "/" name ".edn"))))
        cases (golden "cases")
        want (:system-one (golden "readme"))
        body (seq/json-str (seq/ordered-map [["state" (:readme-state cases)]
                                             ["model" default-model]
                                             ["questions" (:readme-questions cases)]]))
        resp (h {:request-method :post :uri "/v1/systemone" :headers {} :body body})
        ;; the python JSON plus a routing key; probabilities are compared to
        ;; one unit in the fourth decimal, since a value on a rounding
        ;; boundary can land either side under a different BLAS (byte
        ;; identity under Accelerate is the test suite's job)
        got (json/read-str (str/trim (:body resp)))
        approx (fn approx [a b]
                 (cond (and (number? a) (number? b)) (<= (Math/abs (- (double a) (double b))) 1.0001e-4)
                       (and (map? a) (map? b)) (and (= (set (keys a)) (set (keys b)))
                                                    (every? (fn [[k v]] (approx v (get b k))) a))
                       (and (sequential? a) (sequential? b)) (and (= (count a) (count b))
                                                                  (every? true? (map approx a b)))
                       :else (= a b)))
        answer-ok (and (approx (json/read-str want) (dissoc got "routing"))
                       (= "english" (get-in got ["routing" "model"])))
        tok (:tok agent)
        tok-cases (:tok-cases cases)
        tok-golden (:cases (golden "tok"))
        tok-ok (every? (fn [[i c]] (= (get tok-golden (str i)) (tk/encode tok c)))
                       (map-indexed vector tok-cases))
        nfc-ok (= (apply str (repeat 300 "क़"))
                  (tk/nfc (apply str (repeat 300 "क़"))))]
    [["README quickstart answer matches the Python engine" answer-ok
      (when-not answer-ok (str "got " (seq/json-str got)))]
     ["tokenizer reproduces golden/tok.edn" tok-ok nil]
     ["NFC expansion path" nfc-ok nil]]))

(defn -main
  "jolt -M:serve [--data DIR] [--port N] [--host ADDR] [--api-key KEY]
                 [--max-loaded N] [--default-model NAME] [--workflows DIR[:DIR]]
                 [--max-len N] [--head-max-len N]
   Each falls back to an environment variable (LEV_DATA, PORT, LEV_HOST,
   LEV_API_KEY, LAYA_MAX_LOADED, LAYA_DEFAULT_MODEL, LEV_WORKFLOWS,
   LEV_MAX_LEN, LEV_HEAD_MAX_LEN), then to ~/.config/lev/config.edn (:data
   :port :host :api-key :max-loaded :default-model :auto-task-detection
   :workflow-dirs :max-len :head-max-len :checkpoints), then to a default.
   DIR is the data root jolt prepare writes: DIR/ (english),
   DIR/multilingual, DIR/typed-decisions; the default checkpoint is loaded
   at startup, the others on first use.
   --self-test [--golden DIR]: load, verify against golden/, exit 0 or 1."
  [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        arg (fn [flag env key default] (cfg/setting ctx flag env key default))
        data-dir (arg "--data" "LEV_DATA" :data "data")
        rt (router/make-router {:data data-dir
                                :max-loaded (Long/parseLong (str (arg "--max-loaded" "LAYA_MAX_LOADED" :max-loaded "1")))
                                :default (arg "--default-model" "LAYA_DEFAULT_MODEL" :default-model "english")
                                :auto-task-detection (or (true? (get opts "--auto-task-detection"))
                                                         (true? (:auto-task-detection (:config ctx))))
                                ;; cfg/limits resolves CLI > env > config per name; the
                                ;; router keeps the per-checkpoint results
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)})
        log (fn [& xs] (binding [*out* *err*] (apply println "lev:" xs)))
        t0 (System/nanoTime)
        _ (log "loading" (:default rt) "from" (get (:models rt) (:default rt)) "...")
        agent (try (router/load-model rt (:default rt))
                   (catch Exception e
                     (if (= :model-unavailable (:type (ex-data e)))
                       (do (log (ex-message e)) (System/exit 1))
                       (throw e))))
        _ (log (format "%d tensors loaded in %.1fs; max_len %d%s, head_max_len %d"
                       (count (:w agent)) (/ (- (System/nanoTime) t0) 1e9)
                       (:max-len (:cfg agent))
                       (if (not= (:max-len (:cfg agent)) (:trained-max-len agent))
                         (str " (trained " (:trained-max-len agent) ")") "")
                       (:head-max-len (:cfg agent))))]
    (if (get opts "--self-test")
      (let [results (self-test agent (arg "--golden" "LEV_GOLDEN" :golden "golden"))]
        (doseq [[name ok? detail] results]
          (println (if ok? "ok  " "FAIL") name (or detail "")))
        (System/exit (if (every? second results) 0 1)))
      (let [workflows (wf/load-workflows (cfg/workflow-dirs ctx))
            port (Long/parseLong (str (arg "--port" "PORT" :port "8080")))
            host (arg "--host" "LEV_HOST" :host "127.0.0.1")
            api-key (arg "--api-key" "LEV_API_KEY" :api-key nil)
            server (start rt {:port port :host host :api-key api-key :workflows workflows})]
        (log "workflows:" (if (seq workflows) (str/join ", " (sort (keys workflows))) "none"))
        (log "checkpoints:" (str/join ", " (for [n router/names]
                                             (str n (if (router/available? rt n) "" " (not prepared)")))))
        (log (format "listening on http://%s:%d (auth %s)" host (:port server) (if api-key "on" "off")))
        @(promise)))))
