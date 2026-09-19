(ns lev.patterns-test
  "lev.api (von's decide/judge/rate and the question constructors) and
  lev.patterns (von's composable decision patterns: confidence gate,
  route, composite score, two-stage choice) plus the escalation the
  benchmark motivates: a gate that re-asks its unsure questions on a
  thinker. Every test runs on fake agents through a router, so no model
  is loaded."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.api :as api]
            [lev.patterns :as pat]
            [lev.router :as router]
            [lev.sequence :as seq]))

(defn- canned-agent
  "A :laya-shaped agent whose system-one answers from a table {qid p},
  p in answer order ([false true] for a noul)."
  [name table]
  {:kind :canned :name name :table table})

(defmethod ag/system-one* :canned
  [{:keys [name table]} _state questions {:keys [constraints on-infeasible]}]
  (let [prepared (mapv (fn [[qid qdef]]
                         (let [qdef (ag/validate-question qid qdef)]
                           {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
                       questions)
        ;; by question id, else by its instructions (the one-question conveniences
        ;; name their question themselves)
        ps (mapv (fn [{:keys [qid q]}]
                   (or (get table (if (keyword? qid) (clojure.core/name qid) (str qid)))
                       (get table (:ins q))
                       (throw (ex-info (str "canned agent " name ": no answer for " (pr-str qid) " / " (pr-str (:ins q))) {}))))
                 prepared)
        cs (ag/prepare-constraints (map (fn [{:keys [qid qdef]}] [qid qdef]) prepared) constraints)
        sol (when cs (ag/decide-constraints cs (map (fn [{:keys [qid q]} p] [qid q p]) prepared ps) on-infeasible))]
    (seq/ordered-map
     (concat [["model" name]
              ["answers" (seq/ordered-map
                          (map (fn [{:keys [qid q]} p]
                                 [qid (ag/typed-answer q p (count p) (when sol (ag/decided-label cs sol qid)) nil)])
                               prepared ps))]
              ["usage" (array-map "input_tokens" (* 10 (count prepared)) "output_tokens" 0)]]
             (when sol [["constraints" (ag/constraints-report cs sol)]])))))

(def questions
  (array-map
   "intent" (api/choice "What does the customer want?" (array-map "refund" "money back" "help" "a question" "other" nil))
   "is_urgent" (api/noul "Time pressure?")
   "frustration" (api/score "How angry?" ["calm" "annoyed" "furious"])))

(def fast (canned-agent "fast" {"intent" [0.5 0.3 0.2] "is_urgent" [0.45 0.55] "frustration" [0.1 0.8 0.1]
                                "What?" [0.5 0.3 0.2] "Which option best describes the state?" [0.5 0.3 0.2]
                                "Time pressure?" [0.45 0.55] "How angry?" [0.1 0.8 0.1]}))
(def slow (canned-agent "slow" {"intent" [0.05 0.9 0.05] "is_urgent" [0.02 0.98] "frustration" [0.9 0.05 0.05]
                                "What?" [0.05 0.9 0.05]}))

(defn- rt []
  (router/make-router {:loader (fn [name & _] (if (= name "english") fast (throw (ex-info "no" {}))))
                       :thinkers {"slow" {:model "x.gguf"}}
                       :thinker-loader (fn [name _] slow)}))

;; --- lev.api --------------------------------------------------------------------

(deftest question-constructors-build-the-wire-shapes
  (is (= {"type" "choice" "instructions" "pick" "criteria" {"a" "A" "b" nil}}
         (api/choice "pick" {"a" "A" "b" nil})))
  (is (= {"type" "choice" "instructions" "pick" "criteria" {"a" nil "b" nil}}
         (api/choice "pick" ["a" "b"])) "a list of options")
  (is (= {"type" "noul" "instructions" "is it?"} (api/noul "is it?")))
  (is (= {"type" "noul" "instructions" "is it?" "criteria" {"true" "yes" "false" "no"}}
         (api/noul "is it?" {"true" "yes" "false" "no"})))
  (is (= {"type" "score" "instructions" "how much?" "criteria" ["low" "high"]}
         (api/score "how much?" ["low" "high"]))))

(deftest decide-judge-and-rate-on-an-agent-or-a-router
  (testing "decide: the choice answer"
    (let [a (api/decide fast "state" {"refund" "money back" "help" "a question" "other" nil} "What?")]
      (is (= "refund" (get a "choice")))
      (is (= 0.5 (get-in a ["probabilities" "refund"]))))
    (is (= "refund" (get (api/decide fast "state" ["refund" "help" "other"]) "choice")) "a list, the default instructions"))
  (testing "judge: p(true)"
    (is (= 0.55 (api/judge fast "state" "Time pressure?")))
    (is (= 0.55 (api/judge fast "state" "Time pressure?" {"true" "yes" "false" "no"}))))
  (testing "rate: the score answer"
    (let [a (api/rate fast "state" ["calm" "annoyed" "furious"] "How angry?")]
      (is (= 1.0 (get a "score")))
      (is (= {"0" "calm" "1" "annoyed" "2" "furious"} (get a "legend")))))
  (testing "through a router the model is routed (or named), and the routing rides along"
    (let [out (api/system-one (rt) "Refund me" questions)]
      (is (= "fast" (get out "model")))
      (is (= "english" (get-in out ["routing" "model"]))))
    (is (= "help" (get (api/decide (rt) "Refund me" ["refund" "help" "other"] "What?" {:model "slow"}) "choice")))))

;; --- lev.patterns -------------------------------------------------------------

(deftest confidence-gate-splits-on-each-answers-confidence
  (let [{:strs [automatic escalate response]} (pat/confidence-gate fast "s" questions {:threshold 0.5})]
    ;; intent: 1 - H([.5 .3 .2])/ln 3 = 0.06; is_urgent: max(p, 1-p) = 0.55; frustration: 0.4
    (is (= ["is_urgent"] (keys automatic)))
    (is (= ["intent" "frustration"] (keys escalate)))
    (is (= "fast" (get response "model"))))
  (testing "the default threshold is 0.8; 0 lets everything through"
    (is (empty? (get (pat/confidence-gate fast "s" questions {}) "automatic")))
    (is (= 3 (count (get (pat/confidence-gate fast "s" questions {:threshold 0.0}) "automatic")))))
  (testing "a threshold outside [0, 1] is refused"
    (is (thrown-with-msg? Exception #"threshold" (pat/confidence-gate fast "s" questions {:threshold 1.5})))))

(deftest escalation-re-asks-the-unsure-questions-on-a-thinker
  (let [out (pat/escalate (rt) "Refund me" questions {:threshold 0.5 :model "slow"})]
    (testing "the sure answer stays the fast model's, the unsure ones are the thinker's"
      (is (= 0.55 (get-in out ["answers" "is_urgent" "noul"])))
      (is (= "help" (get-in out ["answers" "intent" "choice"])))
      (is (= 0.15 (get-in out ["answers" "frustration" "score"]))))
    (testing "the answers keep their order and the report says what happened"
      (is (= ["intent" "is_urgent" "frustration"] (keys (get out "answers"))))
      (is (= ["model" "answers" "usage" "escalation" "routing"] (keys out)))
      (is (= {"threshold" 0.5 "model" "slow" "escalated" ["intent" "frustration"]
              "usage" {"input_tokens" 20 "output_tokens" 0}}
             (get out "escalation")))
      (is (= "fast" (get out "model")))
      (is (= 30 (get-in out ["usage" "input_tokens"])) "the fast pass's usage; the thinker's is in the report")))
  (testing "nothing to escalate: no second call, an empty list"
    (let [out (pat/escalate (rt) "Refund me" questions {:threshold 0.0 :model "slow"})]
      (is (= [] (get-in out ["escalation" "escalated"])))
      (is (= "refund" (get-in out ["answers" "intent" "choice"])))))
  (testing "constraints decide over the merged answers"
    (let [out (pat/escalate (rt) "Refund me" questions {:threshold 0.5 :model "slow"
                                                         :constraints [["implies" ["intent" "help"] ["is_urgent" false]]]})]
      (is (= ["model" "answers" "usage" "escalation" "constraints" "routing"] (keys out)))
      (is (= "help" (get-in out ["answers" "intent" "decided"])))
      (is (false? (get-in out ["answers" "is_urgent" "decided"])) "0.9 help outweighs 0.55 urgent")
      (is (= ["type" "noul" "decided" "confidence"] (keys (get-in out ["answers" "is_urgent"]))))
      (is (true? (get-in out ["constraints" "feasible"])))))
  (testing "thinking passes through to the thinker; an unknown model is refused before any call"
    (is (thrown-with-msg? Exception #"unknown model" (pat/escalate (rt) "s" questions {:model "gpt"})))))

(deftest route-dispatches-on-the-choice
  (let [q (api/choice "What?" ["refund" "help" "other"])
        handled (atom nil)
        routes {"refund" (fn [a] (reset! handled [:refund (get a "choice")]) :refunded)
                "help" (fn [a] (reset! handled [:help a]) :helped)}]
    (is (= :refunded (pat/route fast "s" q routes {})))
    (is (= [:refund "refund"] @handled))
    (testing "below min-confidence, or without a handler, the default gets the answer"
      (is (= "refund" (get (pat/route fast "s" q routes {:min-confidence 0.9}) "choice")) "no default: the answer itself")
      (is (= :fallback (pat/route fast "s" q routes {:min-confidence 0.9 :default (fn [_] :fallback)})))
      (is (= :fallback (pat/route fast "s" q {} {:default (fn [_] :fallback)}))))
    (is (thrown-with-msg? Exception #"choice" (pat/route fast "s" (api/noul "?") routes {})))))

(deftest composite-score-is-a-weighted-mean-of-normalised-answers
  (let [{:strs [score breakdown response]} (pat/composite-score fast "s" questions {:weights {"frustration" 2.0}})]
    ;; is_urgent 0.55 (w 1), frustration 1.0/2 = 0.5 (w 2); intent excluded
    (is (= (/ (+ 0.55 (* 2 0.5)) 3.0) score))
    (is (= ["is_urgent" "frustration"] (keys breakdown)))
    (is (= {"raw" 1.0 "normalized" 0.5 "weight" 2.0} (get breakdown "frustration")))
    (is (= "fast" (get response "model"))))
  (testing "unnormalised: the weighted sum"
    (is (= (+ 0.55 (* 2 0.5)) (get (pat/composite-score fast "s" questions {:weights {"frustration" 2.0} :normalize false}) "score"))))
  (testing "no numeric questions: 0"
    (is (= 0.0 (get (pat/composite-score fast "s" (select-keys questions ["intent"]) {}) "score")))))

(deftest two-stage-choice-narrows-a-taxonomy
  (let [agent (canned-agent "tx" {"Which broad category best matches the state?" [0.7 0.3]
                                  "Which specific option applies within cloud?" [0.2 0.8]})
        taxonomy (array-map "cloud" (array-map "aws" "Amazon" "gcp" "Google")
                            "database" (array-map "postgres" "PostgreSQL" "redis" "Redis"))
        out (pat/two-stage-choice agent "Postgres replica lag" taxonomy {})]
    (is (= "cloud" (get out "category")))
    (is (= "gcp" (get out "choice")))
    (is (= ["category" "category_confidence" "choice" "choice_confidence" "combined_confidence"] (keys out)))
    (is (= (ag/round4 (* (get out "category_confidence") (get out "choice_confidence"))) (get out "combined_confidence")))))
