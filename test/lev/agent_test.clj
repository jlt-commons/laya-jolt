(ns lev.agent-test
  "End-to-end agent parity: RLAgent.system_one on the README quickstart must
  reproduce golden/readme.edn :system-one to the fourth decimal.

  This guards the whole stack (tokenizer -> encoder -> head -> calibration).
  It is the regression test for the encoder sliding-window radius: with
  local_attention=128 the mask radius is 64 (torch: config.sliding_window =
  local_attention // 2), and a wrong radius silently corrupts every sliding
  layer beyond ~65 tokens."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.sequence :as seq]
            [lev.test-util :as tu :refer [golden-dir data-dir]]
            [lev.workflows :as wf]))

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
          (is (tu/approx= 1.0001e-4 (dissoc want "model") (dissoc got "model" "truncated")) (str "body " body-index)))))))

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
    (testing "equal to the checkpoint's own output (to the fourth decimal)"
      (tu/answers-match (golden-system-one) (seq/json-str out)))
    (testing "typed answers"
      (let [a (get out "answers")]
        (is (= "billing" (get-in a ["department" "choice"])))
        (is (= 1.51 (get-in a ["urgency" "score"])))
        (is (= 0.4312 (get-in a ["churn_risk" "noul"])))
        (is (= 0.0312 (get-in a ["is_phishing" "noul"])))
        (is (= 367 (get-in out ["usage" "input_tokens"])))))))

(deftest answer-shape-is-the-upstream-packages
  ;; Agent.system_one upstream: "action" not "rl_agent", act_probability
  ;; rounded, noul carries a confidence; the model is the agent's name
  (let [out (ag/system-one @agent state questions)
        a (get out "answers")]
    (is (= "encoder" (get out "model")) "loaded alone; through the router it is the checkpoint's name")
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

(deftest sequence-limits-can-be-raised-or-lowered
  ;; the checkpoint's rl_agent_config gives 512 / 192; ~/.config/lev can change
  ;; them per load. RoPE has no position table, so a longer max-len simply
  ;; leaves more room for the state.
  (let [base @agent
        long-state {"body" (apply str (repeat 500 "word "))}
        q {"q" {"type" "noul" "instructions" "Is it long?"}}
        tokens (fn [ag*] (get-in (ag/system-one ag* long-state q) ["usage" "input_tokens"]))]
    (is (= 512 (:max-len (:cfg base))))
    (is (= 512 (tokens base)) "the state is cut to fill max_len")
    (let [wide (ag/with-limits base {:max-len 1024})]
      (is (= 1024 (:max-len (:cfg wide))))
      (is (= 192 (:head-max-len (:cfg wide))) "untouched keys keep the checkpoint's values")
      (is (= 512 (:trained-max-len wide)) "the trained length is remembered for the logs")
      (is (> (tokens wide) 512) "more of the state is read"))
    (let [narrow (ag/with-limits base {:max-len 256 :head-max-len 64})]
      (is (= 256 (tokens narrow))))
    (testing "the same through load-agent"
      (is (= 640 (:max-len (:cfg (ag/load-agent tu/data-dir {:max-len 640}))))))
    (testing "nonsense is refused"
      (doseq [bad [{:max-len 0} {:max-len "big"} {:head-max-len 512 :max-len 512} {:head-max-len 4}]]
        (is (thrown? Exception (ag/with-limits base bad)) (pr-str bad))))))

(deftest constraints-decide-the-questions-jointly
  ;; department: billing (0.9+); urgency 1.51 = level 1 or 2; churn_risk
  ;; 0.4312; is_phishing 0.0312 (system-one-matches-readme)
  (let [plain (ag/system-one @agent state questions)]
    (testing "no constraints, no change: the 4-arity with nil is the 3-arity"
      (is (= (seq/json-str plain) (seq/json-str (ag/system-one @agent state questions nil))))
      (is (= (seq/json-str plain) (seq/json-str (ag/system-one @agent state questions {})))))
    (testing "an empty constraint list still decides: every answer gets `decided` = its own argmax"
      (let [out (ag/system-one @agent state questions {:constraints []})
            a (get out "answers")]
        (is (= ["model" "answers" "usage" "constraints"] (keys out)))
        (is (= ["type" "choice" "decided" "probabilities" "confidence" "action"] (keys (get a "department"))))
        (is (= ["type" "score" "decided" "legend" "probabilities" "confidence" "action"] (keys (get a "urgency"))))
        (is (= ["type" "noul" "decided" "confidence" "action"] (keys (get a "churn_risk"))))
        (is (= "billing" (get-in a ["department" "decided"])))
        (is (= 2 (get-in a ["urgency" "decided"])) "the most probable level, an index into the legend")
        (is (false? (get-in a ["churn_risk" "decided"])))
        (is (false? (get-in a ["is_phishing" "decided"])))
        (is (= (seq/ordered-map [["feasible" true] ["decoder" "independent"] ["exact" true] ["violations" []]])
               (get out "constraints")))
        (is (= (seq/json-str plain)
               (seq/json-str (-> out (dissoc "constraints")
                                 (update "answers" (fn [as] (seq/ordered-map (map (fn [[k v]] [k (dissoc v "decided")]) as)))))))
            "everything else is byte-identical")))
    (testing "a constraint moves `decided`, never the model's own answer"
      (let [out (ag/system-one @agent state questions
                               {:constraints [["implies" ["department" "billing"] ["churn_risk" true]]
                                              ["max-level" "urgency" 1]]})
            a (get out "answers")]
        (is (= "billing" (get-in a ["department" "choice"])))
        (is (= "billing" (get-in a ["department" "decided"])) "0.9 vs 0.43: cheaper to flip churn_risk")
        (is (true? (get-in a ["churn_risk" "decided"])))
        (is (= 0.4312 (get-in a ["churn_risk" "noul"])) "the calibrated probability is untouched")
        (is (= 1 (get-in a ["urgency" "decided"])))
        (is (= 1.51 (get-in a ["urgency" "score"])))
        (is (= "exact" (get-in out ["constraints" "decoder"])))
        (is (true? (get-in out ["constraints" "feasible"])))))
    (testing "infeasible: the fewest violations, reported in canonical form"
      (let [out (ag/system-one @agent state questions
                               {:constraints [[:all-of ["churn_risk" "true"] ["churn_risk" false]]]})]
        (is (false? (get-in out ["constraints" "feasible"])))
        (is (= "min_violations" (get-in out ["constraints" "decoder"])))
        (is (= [["all-of" ["churn_risk" true] ["churn_risk" false]]] (get-in out ["constraints" "violations"])))
        (is (false? (get-in out ["answers" "churn_risk" "decided"])) "the model's own argmax, nothing better")))
    (testing "or raise, with the violations"
      (let [e (try (ag/system-one @agent state questions
                                  {:constraints [["all-of" ["churn_risk" true] ["churn_risk" false]]]
                                   :on-infeasible "raise"})
                   nil (catch Exception e e))]
        (is (= :infeasible (:type (ex-data e))))
        (is (= [["all-of" ["churn_risk" true] ["churn_risk" false]]] (:violations (ex-data e))))))
    (testing "a bad constraint is refused before any inference"
      (let [e (try (ag/system-one @agent state questions {:constraints [["implies" ["department" "legal"] ["churn_risk" true]]]})
                   nil (catch Exception e e))]
        (is (= :invalid-constraint (:type (ex-data e))))
        (is (= 0 (:index (ex-data e))))
        (is (re-find #"\"legal\" is not an option of \"department\"" (ex-message e))))
      (is (thrown-with-msg? Exception #"list" (ag/system-one @agent state questions {:constraints {"a" 1}}))))))

(deftest answers-say-when-the-state-was-cut
  (let [q {"q" {"type" "noul" "instructions" "Is it long?"}}
        short (ag/system-one @agent {"body" "short"} q)
        long-state {"body" (apply str (repeat 600 "word "))}
        cut (ag/system-one @agent long-state q)]
    (testing "a state that fits leaves nothing behind: no key"
      (is (not (contains? short "truncated"))))
    (testing "a state cut to max_len names the questions and the tokens dropped, after usage"
      (is (= ["model" "answers" "usage" "truncated"] (keys cut)))
      (is (= ["q"] (keys (get cut "truncated"))))
      (is (< 50 (get-in cut ["truncated" "q"]) 700))
      (is (= 512 (get-in cut ["usage" "input_tokens"]))))))

(deftest debias-averages-a-choice-over-its-option-rotations
  ;; 20 of 144 authored144 answers change under option rotation (bench/);
  ;; averaging the calibrated probabilities over every rotation is +2.8
  ;; points on english and lowers its ECE, at k forwards per question
  (let [plain (ag/system-one @agent state questions)
        out (ag/system-one @agent state questions {:debias true})
        a (get out "answers")]
    (is (= ["model" "answers" "usage" "debias"] (keys out)))
    (is (= {"department" 4} (get out "debias")) "only choices with 3+ options rotate; score and noul do not")
    (is (= ["billing" "technical" "sales" "other"] (keys (get-in a ["department" "probabilities"]))) "option order is the caller's")
    (is (< (Math/abs (- 1.0 (reduce + (vals (get-in a ["department" "probabilities"]))))) 2e-4))
    (is (not= (get-in plain ["answers" "department" "probabilities"]) (get-in a ["department" "probabilities"])))
    (is (= "billing" (get-in a ["department" "choice"])))
    (is (= (get-in plain ["answers" "urgency"]) (get a "urgency")) "a score is untouched")
    (is (= (get-in plain ["answers" "churn_risk"]) (get a "churn_risk")) "a noul is untouched")
    (testing "usage counts every sequence that ran"
      (is (> (get-in out ["usage" "input_tokens"]) (get-in plain ["usage" "input_tokens"]))))))
