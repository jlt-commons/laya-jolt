(ns laya.agent-test
  "End-to-end agent parity: RLAgent.system_one on the README quickstart must
  reproduce golden/readme.edn :system-one byte-for-byte.

  This guards the whole stack (tokenizer -> encoder -> head -> calibration).
  It is the regression test for the encoder sliding-window radius: with
  local_attention=128 the mask radius is 64 (torch: config.sliding_window =
  local_attention // 2), and a wrong radius silently corrupts every sliding
  layer beyond ~65 tokens."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [laya.agent :as ag]
            [laya.sequence :as seq]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LAYA_DATA") "data"))

(def agent (delay (ag/load-agent data-dir)))

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
  (:system-one (edn/read-string (slurp (str golden-dir "/readme.edn")))))

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
    (testing "byte-for-byte equal to the checkpoint's own output"
      (is (= (golden-system-one) (seq/json-str out))))
    (testing "typed answers"
      (let [a (get out "answers")]
        (is (= "billing" (get-in a ["department" "choice"])))
        (is (= 1.51 (get-in a ["urgency" "score"])))
        (is (= 0.4312 (get-in a ["churn_risk" "noul"])))
        (is (= 0.0312 (get-in a ["is_phishing" "noul"])))
        (is (= 367 (get-in out ["usage" "input_tokens"])))))))
