(ns laya.server-test
  "The HTTP API (laya.server), mirroring TypeSafe's POST /v1/systemone.
  Handler-level tests call the ring handler with request maps; one test
  starts a real server and talks to it with curl."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [laya.agent :as ag]
            [laya.json :as json]
            [laya.sequence :as seq]
            [laya.server :as srv]
            [laya.test-util :as tu]))

(def agent (delay (ag/load-agent tu/data-dir)))

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
    (testing "the body is json.dumps(RLAgent.system_one(...)), byte for byte"
      (is (= (:system-one (tu/read-golden "readme")) (str/trim (:body resp)))))))

(deftest model-field
  (let [h (srv/handler @agent {})
        base (json/read-str (readme-request))
        one-q (seq/ordered-map [["state" "Refund me."]
                                ["questions" (seq/ordered-map [["q" (get-in base ["questions" "churn_risk"])]])]])]
    (testing "echoed when given"
      (let [[st body] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "jev-latest"))))]
        (is (= 200 st))
        (is (= "jev-latest" (get body "model")))
        (is (= ["model" "answers" "usage"] (keys body)))))
    (testing "defaults to the checkpoint's name"
      (let [[st body] (call h (req :post "/v1/systemone" :body (seq/json-str one-q)))]
        (is (= 200 st))
        (is (= "laya-rl-agent" (get body "model")))))
    (testing "must be a string"
      (let [[st body] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" 3))))]
        (is (= 422 st))
        (is (= ["body" "model"] (get-in body ["detail" 0 "loc"])))))))

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
  (let [h (srv/handler @agent {})]
    (is (= 404 (first (call h (req :get "/nope")))))
    (is (= 405 (first (call h (req :get "/v1/systemone")))))
    (is (= 405 (first (call h (req :post "/health")))))
    (let [[st b] (call h (req :get "/health"))]
      (is (= 200 st))
      (is (= {"status" "ok" "model" "laya-rl-agent"} b)))))

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
        (is (= {"status" "ok" "model" "laya-rl-agent"} (json/read-str (str/trim health))))
        (is (= (:system-one (tu/read-golden "readme")) (str/trim answer)))
        (is (= "401" (str/trim denied))))
      (finally
        (srv/stop server)))))
