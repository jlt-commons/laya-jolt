(ns laya.agent-test
  "End-to-end agent parity: RLAgent.system_one on the README quickstart must
  reproduce golden/readme.edn :system-one byte-for-byte.

  This guards the whole stack (tokenizer -> encoder -> head -> calibration).
  It is the regression test for the encoder sliding-window radius: with
  local_attention=128 the mask radius is 64 (torch: config.sliding_window =
  local_attention // 2), and a wrong radius silently corrupts every sliding
  layer beyond ~65 tokens."
  (:require [clojure.string :as str]
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
  "The :system-one JSON embedded in golden/readme.edn (raw, not EDN)."
  []
  (let [r (slurp (str golden-dir "/readme.edn"))
        i (str/index-of r ":system-one ")
        j (str/last-index-of r "}")]
    (str/trim (subs r (+ i (count ":system-one ")) j))))

(deftest sliding-window-is-half-local-attention
  (testing "encoder sliding radius = local_attention // 2 = 64"
    (is (= 64 (:window (:cfg @agent))))))

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
