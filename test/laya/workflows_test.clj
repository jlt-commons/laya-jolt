(ns laya.workflows-test
  "Loading user-defined workflows: a directory of .clj files, each a
  `workflows.<name>` namespace with a `questions` fn and an optional `state`
  builder."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [laya.workflows :as wf]))

(def tmp "target/workflows-test")

(defn- write-workflow! [dir file src]
  (io/make-parents (io/file dir file))
  (spit (str dir "/" file) src))

(defn- fresh! []
  (jolt.host/delete-tree! tmp)
  (io/make-parents (io/file tmp "x")))

(def hello-src
  "(ns workflows.hello-world
  (:require [clojure.string :as str]))
(defn questions
  \"Says hi.\"
  ([] (questions {}))
  ([opts] {\"greet\" {\"type\" \"noul\" \"instructions\" (str \"hi \" (get opts \"who\" \"there\"))}}))
(defn state [input] (str/upper-case (str input)))")

(deftest loads-a-directory-of-workflows
  (fresh!)
  (write-workflow! (str tmp "/a") "hello_world.clj" hello-src)
  (write-workflow! (str tmp "/a") "plain.clj"
                   "(ns workflows.plain)\n(defn questions [] {\"q\" {\"type\" \"noul\" \"instructions\" \"?\"}})")
  (write-workflow! (str tmp "/a") "notes.txt" "not a workflow")
  (let [wfs (wf/load-workflows [(str tmp "/a")])]
    (testing "one entry per .clj, named after the file (underscores -> dashes)"
      (is (= ["hello-world" "plain"] (sort (keys wfs)))))
    (testing "questions with and without options"
      (is (= {"greet" {"type" "noul" "instructions" "hi there"}} (wf/questions (wfs "hello-world"))))
      (is (= {"greet" {"type" "noul" "instructions" "hi bob"}} (wf/questions (wfs "hello-world") {"who" "bob"})))
      (is (= {"q" {"type" "noul" "instructions" "?"}} (wf/questions (wfs "plain") {})))
      (is (thrown-with-msg? Exception #"plain.*options" (wf/questions (wfs "plain") {"ignored" 1}))
          "options to a workflow whose questions fn takes none are an error, not ignored")
      (is (true? (:options? (wfs "hello-world"))))
      (is (false? (:options? (wfs "plain")))))
    (testing "state builder is optional; without one the input is the state"
      (is (= "ABC" (wf/state (wfs "hello-world") "abc")))
      (is (= {"raw" 1} (wf/state (wfs "plain") {"raw" 1}))))
    (testing "the description is the questions docstring"
      (is (= "Says hi." (:doc (wfs "hello-world"))))
      (is (nil? (:doc (wfs "plain")))))
    (testing "each entry remembers where it came from"
      (is (str/ends-with? (:file (wfs "plain")) "a/plain.clj"))))
  (jolt.host/delete-tree! tmp))

(deftest later-directories-override-earlier-ones
  (fresh!)
  (write-workflow! (str tmp "/a") "plain.clj" "(ns workflows.plain)\n(defn questions [] {\"from\" \"a\"})")
  (write-workflow! (str tmp "/b") "plain.clj" "(ns workflows.plain)\n(defn questions [] {\"from\" \"b\"})")
  (is (= {"from" "b"} (wf/questions (get (wf/load-workflows [(str tmp "/a") (str tmp "/b")]) "plain"))))
  (is (= {"from" "a"} (wf/questions (get (wf/load-workflows [(str tmp "/b") (str tmp "/a")]) "plain"))))
  (jolt.host/delete-tree! tmp))

(deftest missing-directories-are-skipped
  (is (= {} (wf/load-workflows ["target/does-not-exist" "target/nor-this"]))))

(deftest bad-workflows-fail-loudly
  (fresh!)
  (testing "no questions fn"
    (write-workflow! (str tmp "/c") "empty.clj" "(ns workflows.empty)")
    (is (thrown-with-msg? Exception #"empty\.clj.*questions" (wf/load-workflows [(str tmp "/c")]))))
  (testing "wrong namespace for the file name"
    (write-workflow! (str tmp "/d") "misnamed.clj" "(ns workflows.other)\n(defn questions [] {})")
    (is (thrown-with-msg? Exception #"misnamed\.clj.*workflows\.misnamed" (wf/load-workflows [(str tmp "/d")]))))
  (testing "a syntax error names the file"
    (write-workflow! (str tmp "/e") "broken.clj" "(ns workflows.broken)\n(defn questions [] {)")
    (is (thrown-with-msg? Exception #"broken\.clj" (wf/load-workflows [(str tmp "/e")]))))
  (jolt.host/delete-tree! tmp))

(deftest bundled-workflows-load
  (testing "the repo's workflows/ directory: email and demo"
    (let [wfs (wf/load-workflows ["workflows"])]
      (is (contains? wfs "email"))
      (is (contains? wfs "demo"))
      (is (= ["category" "is_spam" "is_phishing" "urgency" "needs_reply"]
             (keys (wf/questions (wfs "email")))))
      (is (= {"subject" "S" "body" "b" "from" "a@b.c"}
             (wf/state (wfs "email") {"subject" "S" "body" "b" "from" "a@b.c"})))
      (is (= ["department" "urgency" "churn_risk" "is_phishing"] (keys (wf/questions (wfs "demo"))))))))
