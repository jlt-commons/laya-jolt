(ns workflows.email-test
  "laya email.py parity. golden/email.edn holds clean_email_body in/out
  pairs and json.dumps of email_state / email_questions from the Python
  module, so the port is compared byte for byte."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [lev.workflows :as wf]
            [lev.sequence :as seq]))

;; the bundled email workflow is a file, not a classpath namespace
(wf/load-workflow "workflows/email.clj")
(alias 'email 'workflows.email)

(def golden-dir
  (or (System/getenv "LEV_GOLDEN") "golden"))

(def golden (delay (edn/read-string (slurp (str golden-dir "/email.edn")))))

(deftest clean-email-body-matches-python
  ;; the last pair is the max_chars=100 run of :max-chars-case
  (doseq [[raw want] (butlast (:clean @golden))]
    (is (= want (email/clean-email-body raw)) (pr-str (subs raw 0 (min 60 (count raw))))))
  (testing "max-chars truncates by codepoint"
    (let [raw (:max-chars-case @golden)
          [raw2 want] (last (:clean @golden))]
      (is (= raw raw2))
      (is (= 100 (count want)))
      (is (= want (email/clean-email-body raw 100))))))

(deftest clean-email-body-edge-semantics
  (testing "python whitespace (str.isspace): NBSP and U+001C both strip and count as \\s"
    (is (= "a b" (email/clean-email-body " a b ")))
    (is (= "x\n\ny" (email/clean-email-body "x\n \ny")))
    (is (= "z" (email/clean-email-body "\u001cz"))))
  (testing "quote header only breaks after content has been kept"
    (is (= "On x wrote:\nbody" (email/clean-email-body "> q\nOn x wrote:\nbody")))
    (is (= "body" (email/clean-email-body "body\nOn x wrote:\nold"))))
  (testing "wrote: must be within 300 chars of 'On '"
    (is (= "a" (email/clean-email-body (str "a\nOn " (apply str (repeat 290 "x")) " wrote:\nold"))))
    (is (= (str "a\nOn " (apply str (repeat 310 "x")) " wrote:\nold")
           (email/clean-email-body (str "a\nOn " (apply str (repeat 310 "x")) " wrote:\nold")))))
  (testing "nil body"
    (is (= "" (email/clean-email-body nil)))))

(deftest email-state-matches-python
  (let [[s0 s1 s2 s3] (:states @golden)
        body0 (first (first (:clean @golden)))
        body1 (first (second (:clean @golden)))]
    (is (= s0 (seq/json-str (email/email-state "  Duplicate billing  " body0 :sender "customer@acme.com"))))
    (is (= s1 (seq/json-str (email/email-state nil nil))))
    (is (= s2 (seq/json-str (email/email-state "raw" body1 :clean false))))
    (is (= s3 (seq/json-str (email/email-state "extras" "body text" :sender "a@b.c"
                                               :extra (array-map "priority" 2 "tag" nil
                                                                 "score" 0.5 "flag" true)))))
    (testing "an empty sender is falsy and not recorded"
      (is (= "{\"subject\": \"s\", \"body\": \"b\"}" (seq/json-str (email/email-state "s" "b" :sender "")))))))

(deftest email-questions-match-python
  (is (= (:questions @golden) (seq/json-str (email/email-questions))))
  (is (= (:questions @golden) (seq/json-str (email/email-questions {}))) "empty categories -> defaults")
  (is (= (:questions-custom @golden)
         (seq/json-str (email/email-questions (array-map "refund" "money back" "bug" "it is broken"))))))
