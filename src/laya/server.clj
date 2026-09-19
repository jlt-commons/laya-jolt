(ns laya.server
  "HTTP front for the agent, mirroring the TypeSafe Jev API
  (https://docs.typesafe.ai/api):

    POST /v1/systemone   {\"state\": ..., \"model\": ..., \"questions\": {...}}
                         -> {\"model\": ..., \"answers\": {...}, \"usage\": {...}}
    GET  /health         -> {\"status\": \"ok\", \"model\": \"rl-agent\"}

  Auth is `Authorization: Bearer <key>` when the server is started with an
  :api-key (LAYA_API_KEY); without one every request is accepted. Errors:
  401 for a missing/invalid key, 422 with a FastAPI-style
  {\"detail\": [{\"loc\": [...], \"msg\": ..., \"type\": ...}]} for anything
  wrong with the body, 404/405 for other routes and methods, 413 from the
  adapter when the body exceeds :max-request-bytes.

  As a library: (handler agent opts) is a plain ring handler to mount in
  another app; (start agent opts) / (stop server) run it on
  ring-chez-adapter. As a binary: `jolt build -m laya.server -o laya-server`
  (the `binary` task), then `./laya-server --data data --port 8080`.

  Requests are parsed with laya.json, which keeps object key order: the
  order of options, questions and state fields is model input. Inference
  runs one request at a time behind a lock; the adapter's workers only
  overlap on I/O."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [laya.agent :as ag]
            [laya.json :as json]
            [laya.sequence :as seq]
            [laya.tokenizer :as tk]
            [ring-chez.adapter :as adapter])
  (:gen-class))

(def default-model "rl-agent")

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

;; --- request ---------------------------------------------------------------------

(defn- body-string [body]
  (cond (nil? body) "" (string? body) body :else (slurp body)))

(defn- parse-body
  "The JSON document in the request body; ex-info :invalid-json otherwise."
  [req]
  (let [text (body-string (:body req))]
    (when (str/blank? text)
      (throw (ex-info "json: request body is empty" {:type :invalid-json})))
    (json/read-str text)))

(defn- question-loc [{:keys [qid field]}]
  (cond-> ["body" "questions" (str qid)] field (conj field)))

(defn- check-request
  "Everything wrong with a parsed body, as detail entries. Each question is
  checked with agent/validate-question, so the reasons match the library."
  [body]
  (if-not (map? body)
    [(detail ["body"] "request body must be a JSON object" "type_error")]
    (let [state (get body "state")
          qs (get body "questions")
          model (get body "model")]
      (vec
       (concat
        (when-not (or (string? state) (map? state) (sequential? state))
          [(detail ["body" "state"] "state is required: a string, object or array" "value_error")])
        (when (and (contains? body "model") (not (string? model)))
          [(detail ["body" "model"] "model must be a string" "type_error")])
        (cond
          (not (map? qs)) [(detail ["body" "questions"] "questions is required: an object of question id -> question" "value_error")]
          (empty? qs) [(detail ["body" "questions"] "questions must not be empty" "value_error")]
          :else (keep (fn [[qid qdef]]
                        (try (ag/validate-question qid qdef) nil
                             (catch Exception e
                               (detail (question-loc (ex-data e)) (ex-message e) "value_error"))))
                      qs)))))))

(defn systemone
  "Answer one parsed /v1/systemone body. Inference is serialized on lock."
  [agent lock body]
  (let [details (check-request body)]
    (if (seq details)
      (unprocessable details)
      (let [result (locking lock
                     (ag/system-one agent (get body "state") (get body "questions")))]
        (json-response 200 (assoc result "model" (get body "model" default-model)))))))

(defn- authorized? [req api-key]
  (or (nil? api-key)
      (= (get-in req [:headers "authorization"]) (str "Bearer " api-key))))

(defn handler
  "A ring handler for the API. opts: :api-key (nil = no auth)."
  [agent {:keys [api-key]}]
  (let [lock (Object.)]
    (fn [req]
      (let [method (:request-method req)]
        (case (:uri req)
          "/health"
          (if (= :get method)
            (json-response 200 (seq/ordered-map [["status" "ok"] ["model" default-model]]))
            method-not-allowed)

          "/v1/systemone"
          (cond
            (not= :post method) method-not-allowed
            (not (authorized? req api-key)) unauthorized
            :else
            (try
              (systemone agent lock (parse-body req))
              (catch Exception e
                (case (:type (ex-data e))
                  :invalid-json (unprocessable [(detail ["body"] (ex-message e) "json_invalid")])
                  :invalid-question (unprocessable [(detail (question-loc (ex-data e)) (ex-message e) "value_error")])
                  (throw e)))))

          not-found)))))

;; --- server ----------------------------------------------------------------------

(defn start
  "Run the API on ring-chez-adapter; answers the server handle.
  opts: :port (8080), :host (\"127.0.0.1\"; \"0.0.0.0\" for all interfaces),
  :api-key, :max-request-bytes (4 MiB), plus anything the adapter takes."
  [agent {:keys [port host max-request-bytes]
          :or {port 8080 host "127.0.0.1" max-request-bytes (* 4 1024 1024)}
          :as opts}]
  (adapter/run-server
   (handler agent opts)
   (merge {:on-failure (fn [_ ex]
                         (binding [*out* *err*] (println "laya.server:" (ex-message ex)))
                         (json-response 500 {"detail" "Internal Server Error"}))}
          (dissoc opts :api-key)
          {:port port :host host :max-request-bytes max-request-bytes})))

(defn stop [server]
  (adapter/stop-server server))

(defn self-test
  "Run the README quickstart through the handler and compare with the
  json.dumps(RLAgent.system_one(...)) pinned in <golden-dir>/readme.edn,
  plus the tokenizer paths a release build has miscompiled before.
  Answers [[name ok? detail] ...]. The test suite runs interpreted; this is
  what proves the AOT binary computes the same thing."
  [agent golden-dir]
  (let [h (handler agent {})
        cases (edn/read-string {:readers {'laya/omap seq/ordered-map}}
                               (slurp (str golden-dir "/cases.edn")))
        want (:system-one (edn/read-string (slurp (str golden-dir "/readme.edn"))))
        body (seq/json-str (seq/ordered-map [["state" (:readme-state cases)]
                                             ["model" default-model]
                                             ["questions" (:readme-questions cases)]]))
        resp (h {:request-method :post :uri "/v1/systemone" :headers {} :body body})
        got (str/trim (:body resp))
        tok (:tok agent)
        tok-cases (:tok-cases cases)
        tok-golden (:cases (edn/read-string (slurp (str golden-dir "/tok.edn"))))
        tok-ok (every? (fn [[i c]] (= (get tok-golden (str i)) (tk/encode tok c)))
                       (map-indexed vector tok-cases))
        nfc-ok (= (apply str (repeat 300 "\u0915\u093c"))
                  (tk/nfc (apply str (repeat 300 "\u0958"))))]
    [["README quickstart answer is byte-identical to the Python engine" (= want got)
      (when (not= want got) (str "got " got))]
     ["tokenizer reproduces golden/tok.edn" tok-ok nil]
     ["NFC expansion path" nfc-ok nil]]))

(defn- parse-args
  "--flag value pairs and bare --flags: {\"--port\" \"8080\" \"--self-test\" true}."
  [args]
  (loop [args args out {}]
    (if (empty? args)
      out
      (let [[k v & more] args]
        (if (and v (not (str/starts-with? v "--")))
          (recur more (assoc out k v))
          (recur (rest args) (assoc out k true)))))))

(defn- arg [opts flag env default]
  (let [v (get opts flag)]
    (or (when (string? v) v) (System/getenv env) default)))

(defn -main
  "jolt -M:serve [--data DIR] [--port N] [--host ADDR] [--api-key KEY]
   Environment fallbacks: LAYA_DATA, PORT, LAYA_HOST, LAYA_API_KEY.
   --self-test [--golden DIR]: load, verify against golden/, exit 0 or 1."
  [& args]
  (let [opts (parse-args args)
        data-dir (arg opts "--data" "LAYA_DATA" "data")
        t0 (System/nanoTime)
        _ (binding [*out* *err*] (println "laya: loading" data-dir "..."))
        agent (ag/load-agent data-dir)
        _ (binding [*out* *err*]
            (println (format "laya: %d tensors loaded in %.1fs" (count (:w agent)) (/ (- (System/nanoTime) t0) 1e9))))]
    (if (get opts "--self-test")
      (let [results (self-test agent (arg opts "--golden" "LAYA_GOLDEN" "golden"))]
        (doseq [[name ok? detail] results]
          (println (if ok? "ok  " "FAIL") name (or detail "")))
        (System/exit (if (every? second results) 0 1)))
      (let [port (Long/parseLong (str (arg opts "--port" "PORT" "8080")))
            host (arg opts "--host" "LAYA_HOST" "127.0.0.1")
            api-key (arg opts "--api-key" "LAYA_API_KEY" nil)
            server (start agent {:port port :host host :api-key api-key})]
        (binding [*out* *err*]
          (println (format "laya: listening on http://%s:%d (auth %s)"
                           host (:port server) (if api-key "on" "off"))))
        @(promise)))))
