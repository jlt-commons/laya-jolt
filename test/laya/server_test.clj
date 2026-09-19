(ns laya.server-test
  "The HTTP API (laya.server), mirroring TypeSafe's POST /v1/systemone.
  Handler-level tests call the ring handler with request maps; one test
  starts a real server and talks to it with curl."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [laya.agent :as ag]
            [laya.json :as json]
            [laya.router :as router]
            [laya.sequence :as seq]
            [laya.server :as srv]
            [laya.test-util :as tu]
            [laya.workflows :as wf]))

(def agent tu/agent)
(def workflows (delay (wf/load-workflows ["workflows"])))

(defn- req [method uri & {:keys [body headers]}]
  {:request-method method :uri uri :headers (or headers {}) :body body})

(defn- call
  "[status parsed-body raw-response]"
  [h r]
  (let [resp (h r)]
    [(:status resp) (json/read-str (:body resp)) resp]))

(defn- readme-request []
  (let [cases (tu/read-golden "cases")]
    (seq/json-str (seq/ordered-map [["state" (:readme-state cases)]
                                    ["model" "laya-rl-agent"]
                                    ["questions" (:readme-questions cases)]]))))

(deftest systemone-reproduces-the-python-answer
  (let [h (srv/handler @agent {})
        resp (h (req :post "/v1/systemone" :body (readme-request)))]
    (is (= 200 (:status resp)))
    (is (= "application/json" (get-in resp [:headers "Content-Type"])))
    (testing "the body is json.dumps(Agent.system_one(...)) plus the routing decision"
      (let [body (json/read-str (str/trim (:body resp)))]
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc body "routing")))
        (is (= ["model" "answers" "usage" "routing"] (keys body)))
        (is (= "english" (get-in body ["routing" "model"])))))))

(deftest model-field-selects-the-checkpoint
  (let [h (srv/handler @agent {})
        base (json/read-str (readme-request))
        one-q (seq/ordered-map [["state" "Refund me."]
                                ["questions" (seq/ordered-map [["q" (get-in base ["questions" "churn_risk"])]])]])]
    (testing "absent, or the engine's own name: routed by content, with the decision attached"
      (doseq [body [one-q (assoc one-q "model" "laya-rl-agent") (assoc one-q "model" "rl-agent")]]
        (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str body)))]
          (is (= 200 st))
          (is (= "laya-rl-agent" (get b "model")))
          (is (= ["model" "answers" "usage" "routing"] (keys b)))
          (is (= "english" (get-in b ["routing" "model"])))
          (is (= "English Latin text" (get-in b ["routing" "reason"]))))))
    (testing "a checkpoint name or alias is an explicit choice"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "en"))))]
        (is (= 200 st))
        (is (= "english" (get-in b ["routing" "model"])))
        (is (= "explicit model='en'" (get-in b ["routing" "reason"])))))
    (testing "lang and task route too"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "lang" "en-GB"))))]
        (is (= 200 st))
        (is (= "explicit lang='en-GB'" (get-in b ["routing" "reason"])))))
    (testing "an unknown model is a 422 naming the choices"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "jev-latest"))))]
        (is (= 422 st))
        (is (= ["body" "model"] (get-in b ["detail" 0 "loc"])))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "multilingual"))))
    (testing "a checkpoint that is not prepared is a 503, not a crash"
      (let [h (srv/handler (router/preloaded @agent "english" {:models {"multilingual" "target/not-prepared"}}) {})
            [st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "multilingual"))))]
        (is (= 503 st))
        (is (str/includes? (get b "detail") "multilingual"))))
    (testing "must be strings"
      (doseq [k ["model" "lang" "task"]]
        (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q k 3))))]
          (is (= 422 st) k)
          (is (= ["body" k] (get-in b ["detail" 0 "loc"])) k))))))

(deftest route-without-inference
  (let [boom (fn [name & _] (throw (ex-info (str "must not load " name) {})))
        h (srv/handler (router/make-router {:loader boom}) {})]
    (testing "POST /v1/route answers the decision and loads nothing"
      (let [[st b] (call h (req :post "/v1/route" :body (seq/json-str {"state" {"body" "\u092e\u0941\u091d\u0938\u0947 \u0926\u094b"}})))]
        (is (= 200 st))
        (is (= "multilingual" (get b "model")))
        (is (= "convaiinnovations/laya/multilingual" (get b "repo")))
        (is (= "devanagari" (get-in b ["detection" "script"]))))
      (let [[st b] (call h (req :post "/v1/route" :body (seq/json-str {"state" "hi" "task" "typed_decisions"})))]
        (is (= 200 st))
        (is (= "typed-decisions" (get b "model")))))
    (testing "questions are optional here, state is not"
      (is (= 422 (first (call h (req :post "/v1/route" :body "{}"))))))))

(deftest models-and-workflows-listing
  (let [h (srv/handler @agent {:workflows @workflows})]
    (testing "GET /v1/models: the three checkpoints, which are prepared and loaded"
      (let [[st b] (call h (req :get "/v1/models"))]
        (is (= 200 st))
        (is (= ["english" "multilingual" "typed-decisions"] (keys (get b "models"))))
        (is (= "convaiinnovations/laya" (get-in b ["models" "english" "repo"])))
        (is (true? (get-in b ["models" "english" "loaded"])))
        (is (true? (get-in b ["models" "english" "available"])))
        (is (= {"max_len" 512 "head_max_len" 192} (get-in b ["models" "english" "limits"])))
        (is (= (get-in b ["models" "multilingual" "available"]) (some? (get-in b ["models" "multilingual" "limits"])))
            "limits are known exactly when the checkpoint is prepared")
        (is (= "english" (get b "default")))))
    (testing "GET /v1/workflows: name, description, question ids, whether options are taken"
      (let [[st b] (call h (req :get "/v1/workflows"))]
        (is (= 200 st))
        (is (= ["demo" "email" "guard" "llm-router" "moderation" "triage"] (keys (get b "workflows"))))
        (is (= ["category" "is_spam" "is_phishing" "urgency" "needs_reply"] (get-in b ["workflows" "email" "questions"])))
        (is (true? (get-in b ["workflows" "email" "options"])))
        (is (false? (get-in b ["workflows" "demo" "options"])))
        (is (str/starts-with? (get-in b ["workflows" "email" "description"]) "Email triage"))))))

(deftest workflow-endpoint
  (let [h (srv/handler @agent {:workflows @workflows})
        g (tu/read-golden "email_answers")
        {:keys [body-index state result]} (first (:cases g))
        raw (first (nth (:clean (tu/read-golden "email")) body-index))]
    (testing "POST /v1/workflows/email builds the state, asks its questions, reports both"
      (let [[st b] (call h (req :post "/v1/workflows/email"
                                :body (seq/json-str (seq/ordered-map [["input" (seq/ordered-map [["subject" "Support request"] ["body" raw] ["from" "someone@example.com"]])]]))))
            want (json/read-str result)]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "constraints" "routing" "workflow" "state"] (keys b)))
        (is (= "email" (get b "workflow")))
        (is (= (into {} state) (into {} (get b "state"))) "the cleaned email the model read")
        (is (= (dissoc (get want "answers") "wide")
               (into {} (map (fn [[k a]] [k (dissoc a "decided")]) (get b "answers"))))
            "the golden answers, plus the workflow's constrained decisions")))
    (testing "options reach the workflow; a bare string input is allowed"
      (let [[st b] (call h (req :post "/v1/workflows/email"
                                :body (seq/json-str {"input" "Refund me" "options" {"categories" {"refund" "money back" "other" "else"}}})))]
        (is (= 200 st))
        (is (= ["refund" "other"] (keys (get-in b ["answers" "category" "probabilities"]))))))
    (testing "demo needs no input at all"
      (let [[st b] (call h (req :post "/v1/workflows/demo" :body "{}"))]
        (is (= 200 st))
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc b "routing" "workflow" "state")))))
    (testing "model / lang / task route the workflow request too"
      (let [[st b] (call h (req :post "/v1/workflows/demo" :body (seq/json-str {"model" "english"})))]
        (is (= 200 st))
        (is (= "explicit model='english'" (get-in b ["routing" "reason"])))))
    (testing "unknown workflow is a 404 that lists the known ones"
      (let [[st b] (call h (req :post "/v1/workflows/nope" :body "{}"))]
        (is (= 404 st))
        (is (str/includes? (get b "detail") "email"))))
    (testing "options to a workflow that takes none, or a non-object body, are 422"
      (is (= 422 (first (call h (req :post "/v1/workflows/demo" :body (seq/json-str {"options" {"x" 1}}))))))
      (is (= 422 (first (call h (req :post "/v1/workflows/demo" :body "[1]"))))))))

(deftest bearer-auth
  (let [h (srv/handler @agent {:api-key "s3cret"})
        body (readme-request)]
    (testing "missing or wrong key"
      (doseq [headers [{} {"authorization" "Bearer nope"} {"authorization" "s3cret"} {"authorization" "Basic s3cret"}]]
        (let [[st b] (call h (req :post "/v1/systemone" :body body :headers headers))]
          (is (= 401 st) (pr-str headers))
          (is (= "Missing or invalid API key. Check the Authorization header." (get b "detail"))))))
    (testing "right key"
      (is (= 200 (first (call h (req :post "/v1/systemone" :body body
                                     :headers {"authorization" "Bearer s3cret"}))))))
    (testing "health needs no key"
      (is (= 200 (first (call h (req :get "/health"))))))))

(deftest validation-errors-are-422-with-a-location
  (let [h (srv/handler @agent {})
        post (fn [body] (call h (req :post "/v1/systemone" :body body)))
        q {"type" "noul" "instructions" "ok?"}
        loc (fn [b] (get-in b ["detail" 0 "loc"]))]
    (testing "malformed json"
      (let [[st b] (post "{\"state\": ")]
        (is (= 422 st))
        (is (= ["body"] (loc b)))
        (is (str/starts-with? (get-in b ["detail" 0 "msg"]) "json:"))))
    (testing "empty body"
      (is (= 422 (first (post "")))))
    (testing "not an object"
      (is (= ["body"] (loc (second (post "[1, 2]"))))))
    (testing "state"
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"questions" {"q" q}}))))))
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"state" nil "questions" {"q" q}}))))))
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"state" 3 "questions" {"q" q}})))))))
    (testing "questions"
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s"}))))))
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s" "questions" {}}))))))
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s" "questions" [q]})))))))
    (testing "each bad question is reported with its id and field"
      (let [[st b] (post (seq/json-str (seq/ordered-map
                                        [["state" "s"]
                                         ["questions" (seq/ordered-map
                                                       [["ok" q]
                                                        ["bad_type" {"type" "yesno" "instructions" "x"}]
                                                        ["bad_crit" {"type" "score" "instructions" "x" "criteria" ["one"]}]
                                                        ["no_ins" {"type" "choice" "criteria" ["a" "b"]}]])]])))]
        (is (= 422 st))
        (is (= [["body" "questions" "bad_type" "type"]
                ["body" "questions" "bad_crit" "criteria"]
                ["body" "questions" "no_ins" "instructions"]]
               (map #(get % "loc") (get b "detail"))))
        (is (every? #(= "value_error" (get % "type")) (get b "detail")))))
    (testing "options that cannot fit"
      (let [many (seq/ordered-map (map (fn [i] [(str "option-" i) nil]) (range 200)))
            [st b] (post (seq/json-str {"state" "s" "questions" {"wide" {"type" "choice" "instructions" "x" "criteria" many}}}))]
        (is (= 422 st))
        (is (= ["body" "questions" "wide" "criteria"] (loc b)))))))

(deftest routing
  (let [h (srv/handler @agent {:workflows @workflows})]
    (is (= 404 (first (call h (req :get "/nope")))))
    (is (= 405 (first (call h (req :get "/v1/systemone")))))
    (is (= 405 (first (call h (req :post "/health")))))
    (is (= 405 (first (call h (req :get "/v1/workflows/email")))))
    (is (= 405 (first (call h (req :post "/v1/models")))))
    (let [[st b] (call h (req :get "/health"))]
      (is (= 200 st))
      (is (= {"status" "ok" "model" "laya-rl-agent" "loaded" ["english"]
              "workflows" ["demo" "email" "guard" "llm-router" "moderation" "triage"]} b)))))

(deftest wire-order-is-preserved
  (testing "14 options and 9 questions keep their JSON order end to end"
    (let [h (srv/handler @agent {})
          opts ["billing" "refund" "bug" "outage" "login" "pricing" "demo"
                "hiring" "payroll" "legal" "shipping" "returns" "feedback" "other"]
          q (str "{\"type\": \"choice\", \"instructions\": \"Pick the closest topic.\", \"criteria\": {"
                 (str/join ", " (map #(str "\"" % "\": null") opts)) "}}")
          body (str "{\"state\": \"Refund the duplicate March invoice.\", \"questions\": {"
                    (str/join ", " (map #(str "\"q" % "\": " q) (range 9))) "}}")
          [st b] (call h (req :post "/v1/systemone" :body body))]
      (is (= 200 st))
      (is (= (map #(str "q" %) (range 9)) (keys (get b "answers"))))
      (is (= opts (keys (get-in b ["answers" "q0" "probabilities"]))))
      (is (= (get-in b ["answers" "q0"]) (get-in b ["answers" "q8"]))))))

(deftest real-http-roundtrip
  (let [server (srv/start @agent {:port 18321 :host "127.0.0.1" :api-key "k"})
        base "http://127.0.0.1:18321"]
    (try
      (let [curl (fn [args] (jolt.host/sh-out (str "curl -s " args)))
            health (curl (str base "/health"))
            body (str/replace (readme-request) "'" "'\\''")
            answer (curl (str "-X POST -H 'Content-Type: application/json' -H 'Authorization: Bearer k' -d '"
                             body "' " base "/v1/systemone"))
            denied (curl (str "-o /dev/null -w '%{http_code}' -X POST -d '{}' " base "/v1/systemone"))]
        (is (= {"status" "ok" "model" "laya-rl-agent" "loaded" ["english"] "workflows" []}
               (json/read-str (str/trim health))))
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc (json/read-str (str/trim answer)) "routing")))
        (is (= "401" (str/trim denied))))
      (finally
        (srv/stop server)))))

(deftest constraints-on-the-wire
  (let [h (srv/handler @agent {:workflows @workflows})
        base (json/read-str (readme-request))
        post (fn [path body] (call h (req :post path :body (seq/json-str body))))]
    (testing "constraints decide jointly; the report rides along after usage"
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" [["implies" ["department" "billing"] ["churn_risk" true]]]))]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "constraints" "routing"] (keys b)))
        (is (= "billing" (get-in b ["answers" "department" "decided"])))
        (is (true? (get-in b ["answers" "churn_risk" "decided"])))
        (is (= 0.4312 (get-in b ["answers" "churn_risk" "noul"])))
        (is (= {"feasible" true "decoder" "exact" "exact" true "violations" []} (get b "constraints")))))
    (testing "without them the body is exactly what it was"
      (let [[_ b] (post "/v1/systemone" base)]
        (is (= ["model" "answers" "usage" "routing"] (keys b)))
        (is (not (contains? (get-in b ["answers" "department"]) "decided")))))
    (testing "a bad constraint is a 422 at its index, before any inference"
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" [["not" ["churn_risk" true]] ["implies" ["department" "legal"] ["churn_risk" true]]]))]
        (is (= 422 st))
        (is (= ["body" "constraints" 1] (get-in b ["detail" 0 "loc"])))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "\"legal\" is not an option of \"department\""))
        (is (= "value_error" (get-in b ["detail" 0 "type"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" {"implies" 1}))]
        (is (= 422 st))
        (is (= ["body" "constraints"] (get-in b ["detail" 0 "loc"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "on_infeasible" "explode"))]
        (is (= 422 st))
        (is (= ["body" "on_infeasible"] (get-in b ["detail" 0 "loc"])))))
    (testing "on_infeasible: min_violations reports, raise is a 422 with the violations"
      (let [contradiction [["all-of" ["churn_risk" true] ["churn_risk" false]]]
            [st b] (post "/v1/systemone" (assoc base "constraints" contradiction))]
        (is (= 200 st))
        (is (false? (get-in b ["constraints" "feasible"])))
        (is (= contradiction (get-in b ["constraints" "violations"])))
        (let [[st b] (post "/v1/systemone" (assoc base "constraints" contradiction "on_infeasible" "raise"))]
          (is (= 422 st))
          (is (= ["body" "constraints"] (get-in b ["detail" 0 "loc"])))
          (is (= "infeasible" (get-in b ["detail" 0 "type"])))
          (is (= contradiction (get-in b ["detail" 0 "violations"]))))))
    (testing "a workflow's own constraints apply, and the request's add to them"
      (let [[st b] (post "/v1/workflows/email" {"input" {"subject" "WIN A PRIZE" "body" "You have been selected! Click here to claim your reward now!!!"}})]
        (is (= 200 st))
        (is (contains? (get-in b ["answers" "is_spam"]) "decided"))
        (is (true? (get-in b ["constraints" "feasible"])))
        (when (true? (get-in b ["answers" "is_spam" "decided"]))
          (is (false? (get-in b ["answers" "needs_reply" "decided"])) "spam implies no reply")))
      (let [[st b] (post "/v1/workflows/email" {"input" "Refund me" "constraints" [["min-level" "urgency" 2]]})]
        (is (= 200 st))
        (is (= 2 (get-in b ["answers" "urgency" "decided"]))))
      (let [[st b] (post "/v1/workflows/email" {"input" "Refund me" "constraints" [["min-level" "urgency" 5]]})]
        (is (= 422 st))
        (is (= ["body" "constraints" 0] (get-in b ["detail" 0 "loc"])) "indexed among the request's own")))
    (testing "the listing shows a workflow's constraints"
      (let [[_ b] (call h (req :get "/v1/workflows"))]
        (is (= [["implies" ["is_spam" true] ["needs_reply" false]]
                ["implies" ["is_phishing" true] ["needs_reply" false]]]
               (get-in b ["workflows" "email" "constraints"])))
        (is (= [] (get-in b ["workflows" "demo" "constraints"])))))))
