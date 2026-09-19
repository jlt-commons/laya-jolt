(ns laya.run-test
  "jolt -M:run <workflow> [input]: the CLI over laya.workflows."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [laya.run :as run]
            [laya.sequence :as seq]
            [laya.test-util :as tu]))

(def agent tu/agent)

(deftest demo-is-the-quickstart
  (testing "run demo prints the golden README answer"
    (let [out (run/run-workflow @agent ["workflows"] "demo" nil nil)]
      (is (= (:system-one (tu/read-golden "readme")) (seq/json-str out))))))

(deftest email-workflow-from-json-input
  (let [g (tu/read-golden "email_answers")
        {:keys [body-index result]} (first (:cases g))
        raw (first (nth (:clean (tu/read-golden "email")) body-index))
        input (seq/json-str (seq/ordered-map [["subject" "Support request"] ["body" raw] ["from" "someone@example.com"]]))
        out (run/run-workflow @agent ["workflows"] "email" input nil)
        want (json/read-str result)
        got (json/read-str (seq/json-str out))]
    (testing "same answers as the golden email fan-out, minus the extra wide question"
      (is (= (dissoc (get want "answers") "wide") (get got "answers"))))))

(deftest runner-errors
  (testing "unknown workflow lists the known ones"
    (is (thrown-with-msg? Exception #"nope.*demo.*email"
                          (run/run-workflow @agent ["workflows"] "nope" nil nil))))
  (testing "options go to the workflow"
    (let [out (run/run-workflow @agent ["workflows"] "email"
                                "{\"subject\": \"s\", \"body\": \"Refund me\"}"
                                "{\"categories\": {\"refund\": \"money back\", \"other\": \"else\"}}")]
      (is (= ["refund" "other"] (keys (get-in out ["answers" "category" "probabilities"]))))))
  (testing "@file reads the input from a file"
    (spit "target/run-input.json" "{\"subject\": \"s\", \"body\": \"b\"}")
    (let [out (run/run-workflow @agent ["workflows"] "email" "@target/run-input.json" nil)]
      (is (= ["category" "is_spam" "is_phishing" "urgency" "needs_reply"] (keys (get out "answers")))))
    (jolt.host/delete-tree! "target/run-input.json")))

(deftest list-workflows
  (let [text (with-out-str (run/list-workflows ["workflows"]))]
    (is (str/includes? text "demo"))
    (is (str/includes? text "email"))
    (is (str/includes? text "Email triage") "descriptions come from the questions docstring")))
