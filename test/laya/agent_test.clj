(ns laya.agent-test
  "End-to-end agent parity: RLAgent.system_one on the README quickstart must
  reproduce golden/readme.edn :system-one byte-for-byte.

  This guards the whole stack (tokenizer -> encoder -> head -> calibration).
  It is the regression test for the encoder sliding-window radius: with
  local_attention=128 the mask radius is 64 (torch: config.sliding_window =
  local_attention // 2), and a wrong radius silently corrupts every sliding
  layer beyond ~65 tokens."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [laya.agent :as ag]
            [laya.sequence :as seq]
            [laya.test-util :as tu :refer [golden-dir data-dir]]
            [laya.workflows :as wf]))

;; the bundled email workflow is a file, not a classpath namespace
(wf/load-workflow "workflows/email.clj")
(alias 'email 'workflows.email)

(def agent tu/agent)

(def state
  (array-map
   "from" "customer@acme.com"
   "subject" "Duplicate billing on March invoice #4411"
   "body" "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan."))

(def questions
  (array-map
   "department" {:type "choice"
                 :instructions "Which department should handle this email?"
                 :criteria (array-map "billing" "invoices, payments, refunds"
                                      "technical" "bugs, outages, integrations"
                                      "sales" "pricing, contracts, demos"
                                      "other" "everything else")}
   "urgency" {:type "score"
              :instructions "How urgent is this request?"
              :criteria ["not urgent" "soon" "critical deadline or blocking issue"]}
   "churn_risk" {:type "noul"
                 :instructions "Does the user threaten to cancel or switch to a competitor?"}
   "is_phishing" {:type "noul"
                  :instructions "Is this email a phishing or scam attempt?"}))

(defn- golden-system-one
  "json.dumps(RLAgent.system_one(...)) as captured in golden/readme.edn."
  []
  (:system-one (tu/read-golden "readme")))

(deftest sliding-window-is-half-local-attention
  (testing "encoder sliding radius = local_attention // 2 = 64"
    (is (= 64 (:window (:cfg @agent))))))

(deftest to-internal-mirrors-python
  (testing "list criteria become {c: None}; non-string instructions are json.dumps'd (ascii)"
    (is (= {:t "choice" :ins "pick" :crit (array-map "a" nil "b" nil)}
           (ag/to-internal {:type "choice" :instructions "pick" :criteria ["a" "b"]})))
    (is (= {:t "choice" :ins "pick" :crit (array-map "a" nil "b" nil)}
           (ag/to-internal {:type "choice" :instructions "pick" :criteria '("a" "b")})))
    (is (= "{\"rule\": \"caf\\u00e9\"}"
           (:ins (ag/to-internal {:type "noul" :instructions {"rule" "café"}})))))
  (testing "string keys (a question map parsed from JSON) are accepted too"
    (is (= {:t "score" :ins "how" :crit ["a" "b"]}
           (ag/to-internal {"type" "score" "instructions" "how" "criteria" ["a" "b"]})))))

(deftest options-must-fit-in-head
  (testing "ValueError parity: more options than head_max_len/max_len can hold"
    (let [many (seq/ordered-map (map (fn [i] [(str "option-" i) nil]) (range 200)))
          qs (array-map "q" {:type "choice" :instructions "pick one" :criteria many})]
      (is (thrown-with-msg? Exception #"do not fit in head_max_len"
                            (ag/system-one @agent "state" qs))))))

(deftest malformed-questions-are-rejected
  (testing "python fails on these with KeyError/AttributeError; we say why"
    (let [run (fn [qdef] (ag/system-one @agent "state" {"q" qdef}))]
      (is (thrown-with-msg? Exception #"unknown type" (run {:type "bool" :instructions "x"})))
      (is (thrown-with-msg? Exception #"instructions" (run {:type "noul"})))
      (is (thrown-with-msg? Exception #"criteria" (run {:type "choice" :instructions "x"})))
      (is (thrown-with-msg? Exception #"criteria" (run {:type "choice" :instructions "x" :criteria {}})))
      (is (thrown-with-msg? Exception #"criteria" (run {:type "choice" :instructions "x" :criteria "billing"})))
      (is (thrown-with-msg? Exception #"at least 2" (run {:type "score" :instructions "x" :criteria ["only"]})))
      (is (thrown-with-msg? Exception #"criteria" (run {:type "score" :instructions "x" :criteria {"a" "b"}})))
      (is (thrown-with-msg? Exception #"criteria" (run {:type "noul" :instructions "x" :criteria ["yes" "no"]}))))
    (testing "the failing question id is reported"
      (is (thrown-with-msg? Exception #"\"bad_one\""
                            (ag/system-one @agent "state" {"bad_one" {:type "nope" :instructions "x"}}))))))

(deftest email-fanout-matches-python
  (testing "email_state + email_questions (+ a 14-option choice) end to end, two emails"
    (let [g (tu/read-golden "email_answers")
          bodies (mapv first (:clean (tu/read-golden "email")))
          qs (:questions g)]
      (is (= 6 (count qs)))
      (doseq [{:keys [body-index state result]} (:cases g)]
        (let [st (email/email-state "Support request" (nth bodies body-index) :sender "someone@example.com")
              want (json/read-str result)
              got (json/read-str (seq/json-str (ag/system-one @agent st qs)))]
          (is (= state st) "email-state rebuilds the Python state")
          (is (= (get want "usage") (get got "usage")) "token count")
          (is (= (into {} (map (fn [[k a]] [k [(get a "type") (get a "choice")]]) (get want "answers")))
                 (into {} (map (fn [[k a]] [k [(get a "type") (get a "choice")]]) (get got "answers"))))
              "types and choices")
          ;; Python's calibrated softmax runs in float32, ours in doubles: a
          ;; probability within ~1e-7 of a 4-decimal boundary may round to
          ;; the neighbouring digit, so allow one unit in the last place.
          (is (tu/approx= 1.0001e-4 want got) (str "body " body-index)))))))

(deftest wide-choice-keeps-option-order
  (testing "past 8 options (and 8 questions) the answer maps must still follow input order"
    (let [opts ["billing" "refund" "bug" "outage" "login" "pricing" "demo"
                "hiring" "payroll" "legal" "shipping" "returns" "feedback" "other"]
          q {:type "choice" :instructions "Pick the closest topic." :criteria opts}
          qs (seq/ordered-map (map (fn [i] [(str "q" i) q]) (range 9)))
          out (ag/system-one @agent "Refund the duplicate March invoice, please." qs)]
      (is (= (map #(str "q" %) (range 9)) (keys (get out "answers"))))
      (is (= opts (keys (get-in out ["answers" "q0" "probabilities"]))))
      (is (= 14 (count (get-in out ["answers" "q0" "probabilities"]))))
      (is (contains? (set opts) (get-in out ["answers" "q0" "choice"])))
      (is (= (get-in out ["answers" "q0"]) (get-in out ["answers" "q8"])) "same question, same answer"))))

(deftest system-one-matches-readme
  (let [out (ag/system-one @agent state questions)]
    (testing "equal to the checkpoint's own output (byte-for-byte under Accelerate)"
      (tu/answers-match (golden-system-one) (seq/json-str out)))
    (testing "typed answers"
      (let [a (get out "answers")]
        (is (= "billing" (get-in a ["department" "choice"])))
        (is (= 1.51 (get-in a ["urgency" "score"])))
        (is (= 0.4312 (get-in a ["churn_risk" "noul"])))
        (is (= 0.0312 (get-in a ["is_phishing" "noul"])))
        (is (= 367 (get-in out ["usage" "input_tokens"])))))))

(deftest answer-shape-is-laya-0-3
  ;; laya 0.3.0 (Agent.system_one): "action" not "rl_agent", act_probability
  ;; rounded, noul carries a confidence, model is "laya-rl-agent"
  (let [out (ag/system-one @agent state questions)
        a (get out "answers")]
    (is (= "laya-rl-agent" (get out "model")))
    (is (= ["type" "choice" "probabilities" "confidence" "action"] (keys (get a "department"))))
    (is (= ["type" "score" "legend" "probabilities" "confidence" "action"] (keys (get a "urgency"))))
    (is (= ["type" "noul" "confidence" "action"] (keys (get a "churn_risk"))))
    (testing "noul confidence = round(max(p, 1-p), 4)"
      (let [p (get-in a ["churn_risk" "noul"])]
        (is (= (/ (Math/round (* 1e4 (max p (- 1 p)))) 1e4) (get-in a ["churn_risk" "confidence"])))))
    (testing "act_probability is rounded to 4 decimals"
      (doseq [[_ ans] a]
        (let [ap (get-in ans ["action" "act_probability"])]
          (is (= ap (/ (Math/round (* 1e4 ap)) 1e4))))))))

(deftest calibration-guards
  (testing "confidence is clipped to [0, 1]"
    (is (= 1.0 (ag/confidence-from-probs [1.0 0.0] 2)))
    (is (= 0.0 (ag/confidence-from-probs [0.5 0.5] 2)))
    (is (= 1.0 (ag/confidence-from-probs [1.0] 1)))
    (is (<= 0.0 (ag/confidence-from-probs [0.25 0.25 0.25 0.25] 4) 1e-12)))
  (testing "a zero temperature is floored at 1e-3 instead of dividing by zero"
    (let [cold (update @agent :cfg assoc :temperature [0.0 0.0 0.0] :temperature-by-options {})
          out (ag/system-one cold state (select-keys questions ["churn_risk"]))
          p (get-in out ["answers" "churn_risk" "noul"])]
      (is (number? p))
      (is (not (Double/isNaN p)))
      (is (contains? #{0.0 1.0} p) "a floored temperature saturates the softmax"))))
