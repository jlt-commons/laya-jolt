(ns lev.constraints-test
  "Constrained multi-question decoding (lev.constraints), the torch-free
  port of gliner2.classification: the constraint data language, Kleene
  evaluation over partial assignments, and the decoders (independent,
  exact with branch and bound, beam, min-violations) proved against brute
  force on random problems."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.constraints :as c]))

(def schema
  "What agent/system-one builds from a question map: every label a
  question can decide on, in option order."
  (c/schema (array-map
             "category" {:type "choice" :criteria (array-map "billing" "invoices" "technical" "bugs" "other" "else")}
             "urgency" {:type "score" :criteria ["no time pressure" "needs attention soon" "blocking"]}
             "is_spam" {:type "noul"}
             "needs_reply" {:type "noul" :criteria {"true" "expects a reply"}})))

(deftest schema-lists-the-labels-in-option-order
  (is (= ["billing" "technical" "other"] (get-in schema ["category" :labels])))
  (is (= [0 1 2] (get-in schema ["urgency" :labels])))
  (is (= ["no time pressure" "needs attention soon" "blocking"] (get-in schema ["urgency" :legend])))
  (is (= [true false] (get-in schema ["is_spam" :labels])))
  (is (= "noul" (get-in schema ["needs_reply" :type]))))

;; --- parsing ---------------------------------------------------------------------

(deftest refs-name-a-question-and-one-of-its-labels
  (testing "choice by option, score by level index or legend text, noul by boolean"
    (is (= [{:op :ref :q "category" :i 1 :label "technical"}] (c/parse schema [["category" "technical"]])))
    (is (= [{:op :ref :q "urgency" :i 2 :label 2}] (c/parse schema [["urgency" 2]])))
    (is (= [{:op :ref :q "urgency" :i 2 :label 2}] (c/parse schema [["urgency" "blocking"]])))
    (is (= [{:op :ref :q "is_spam" :i 0 :label true}] (c/parse schema [["is_spam" true]])))
    (is (= [{:op :ref :q "is_spam" :i 1 :label false}] (c/parse schema [["is_spam" "false"]]))))
  (testing "a bad ref says which constraint and why"
    (doseq [[bad msg] [[["nope" true] #"unknown question"]
                       [["category" "refund"] #"not an option of"]
                       [["urgency" 3] #"level"]
                       [["urgency" "later"] #"level"]
                       [["is_spam" "maybe"] #"true or false"]
                       [["is_spam"] #"\[question label\]"]
                       ["is_spam" #"\[question label\]"]
                       [42 #"vector"]]]
      (let [e (try (c/parse schema [["is_spam" true] bad]) nil (catch Exception e e))]
        (is (some? e) (pr-str bad))
        (is (= :invalid-constraint (:type (ex-data e))) (pr-str bad))
        (is (= 1 (:index (ex-data e))) "the index of the offending constraint")
        (is (re-find msg (ex-message e)) (str (pr-str bad) " -> " (ex-message e)))))))

(deftest operators-take-strings-keywords-or-underscores
  (let [want [{:op :implies :a {:op :ref :q "is_spam" :i 0 :label true}
               :b {:op :ref :q "needs_reply" :i 1 :label false}}]]
    (is (= want (c/parse schema [["implies" ["is_spam" true] ["needs_reply" false]]])))
    (is (= want (c/parse schema [[:implies ["is_spam" true] ["needs_reply" false]]])))
    (is (= want (c/parse schema [["IMPLIES" ["is_spam" true] ["needs_reply" false]]]))))
  (is (= [{:op :min-level :q "urgency" :i 1}] (c/parse schema [["min_level" "urgency" 1]])))
  (is (= [{:op :min-level :q "urgency" :i 1}] (c/parse schema [[:min-level "urgency" "needs attention soon"]])))
  (testing "every operator"
    (let [spam {:op :ref :q "is_spam" :i 0 :label true}
          reply {:op :ref :q "needs_reply" :i 0 :label true}]
      (is (= [{:op :not :x spam}] (c/parse schema [["not" ["is_spam" true]]])))
      (is (= [{:op :and :xs [spam reply]}] (c/parse schema [["all-of" ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :or :xs [spam reply]}] (c/parse schema [["any-of" ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :iff :a spam :b reply}] (c/parse schema [["iff" ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :excludes :a spam :b reply}] (c/parse schema [["excludes" ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :count :min 1 :max 1 :xs [spam reply]}] (c/parse schema [["exactly-one-of" ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :count :min 1 :max nil :xs [spam reply]}] (c/parse schema [["at-least" 1 ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :count :min 0 :max 1 :xs [spam reply]}] (c/parse schema [["at-most" 1 ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :count :min 2 :max 2 :xs [spam reply]}] (c/parse schema [["exactly" 2 ["is_spam" true] ["needs_reply" true]]])))
      (is (= [{:op :max-level :q "urgency" :i 1}] (c/parse schema [["max-level" "urgency" 1]])))
      (is (= [{:op :ref :q "urgency" :i 1 :label 1}] (c/parse schema [["at-level" "urgency" 1]])))
      (is (= [{:op :and :xs [{:op :min-level :q "urgency" :i 0} {:op :max-level :q "urgency" :i 1}]}]
             (c/parse schema [["between-level" "urgency" 0 1]])))))
  (testing "arity and argument errors"
    (doseq [bad [["not"] ["not" ["is_spam" true] ["is_spam" false]] ["implies" ["is_spam" true]]
                 ["all-of"] ["at-least" "one" ["is_spam" true]] ["at-least" -1 ["is_spam" true]] ["at-least" 1]
                 ["min-level" "category" "billing"] ["min-level" "urgency"] ["between-level" "urgency" 2 1]
                 ["frobnicate" ["is_spam" true]] []]]
      (let [e (try (c/parse schema [bad]) nil (catch Exception e e))]
        (is (= :invalid-constraint (:type (ex-data e))) (pr-str bad))
        (is (= 0 (:index (ex-data e))) (pr-str bad)))))
  (testing "the whole thing must be a list"
    (is (thrown-with-msg? Exception #"list" (c/parse schema {"implies" 1})))))

(deftest canonical-form-round-trips
  (let [given [[:implies ["is_spam" "true"] ["needs_reply" false]]
               ["at_most" 1 ["is_spam" true] ["urgency" "blocking"]]
               ["between-level" "urgency" "no time pressure" 1]]
        canon (mapv c/canonical (c/parse schema given))]
    (is (= [["implies" ["is_spam" true] ["needs_reply" false]]
            ["at-most" 1 ["is_spam" true] ["urgency" 2]]
            ["all-of" ["min-level" "urgency" 0] ["max-level" "urgency" 1]]]
           canon))
    (is (= (c/parse schema given) (c/parse schema canon)) "canonical form parses back to the same AST")))

;; --- Kleene evaluation ------------------------------------------------------------

(defn- ev [data assign]
  (c/evaluate (first (c/parse schema [data])) schema assign))

(deftest refs-are-undetermined-until-the-question-is-decided
  (is (nil? (ev ["is_spam" true] {})))
  (is (true? (ev ["is_spam" true] {"is_spam" 0})))
  (is (false? (ev ["is_spam" true] {"is_spam" 1})))
  (is (true? (ev ["category" "other"] {"category" 2}))))

(deftest boolean-connectives-are-three-valued
  (let [spam ["is_spam" true] reply ["needs_reply" true]]
    (testing "not"
      (is (nil? (ev ["not" spam] {})))
      (is (false? (ev ["not" spam] {"is_spam" 0}))))
    (testing "and: false dominates, then nil"
      (is (false? (ev ["all-of" spam reply] {"is_spam" 1})))
      (is (nil? (ev ["all-of" spam reply] {"is_spam" 0})))
      (is (true? (ev ["all-of" spam reply] {"is_spam" 0 "needs_reply" 0}))))
    (testing "or: true dominates, then nil"
      (is (true? (ev ["any-of" spam reply] {"is_spam" 0})))
      (is (nil? (ev ["any-of" spam reply] {"is_spam" 1})))
      (is (false? (ev ["any-of" spam reply] {"is_spam" 1 "needs_reply" 1}))))
    (testing "implies = not a or b"
      (is (true? (ev ["implies" spam reply] {"is_spam" 1})) "false antecedent settles it")
      (is (true? (ev ["implies" spam reply] {"needs_reply" 0})) "true consequent settles it")
      (is (nil? (ev ["implies" spam reply] {"is_spam" 0})))
      (is (false? (ev ["implies" spam reply] {"is_spam" 0 "needs_reply" 1}))))
    (testing "iff needs both sides"
      (is (nil? (ev ["iff" spam reply] {"is_spam" 0})))
      (is (true? (ev ["iff" spam reply] {"is_spam" 1 "needs_reply" 1})))
      (is (false? (ev ["iff" spam reply] {"is_spam" 0 "needs_reply" 1}))))
    (testing "excludes = not (a and b)"
      (is (true? (ev ["excludes" spam reply] {"is_spam" 1})))
      (is (nil? (ev ["excludes" spam reply] {"is_spam" 0})))
      (is (false? (ev ["excludes" spam reply] {"is_spam" 0 "needs_reply" 0}))))))

(deftest counting-nodes-bound-the-count-from-both-sides
  (let [xs [["is_spam" true] ["needs_reply" true] ["category" "billing"]]]
    (testing "exactly-one-of"
      (let [c (into ["exactly-one-of"] xs)]
        (is (nil? (ev c {})))
        (is (false? (ev c {"is_spam" 0 "needs_reply" 0})) "two trues: violated before the third decides")
        (is (nil? (ev c {"is_spam" 0})) "one true, two open")
        (is (true? (ev c {"is_spam" 0 "needs_reply" 1 "category" 1})))
        (is (false? (ev c {"is_spam" 1 "needs_reply" 1 "category" 1})))))
    (testing "at-least k / at-most k / exactly k"
      (is (true? (ev (into ["at-least" 1] xs) {"category" 0})) "one true is enough")
      (is (false? (ev (into ["at-least" 2] xs) {"is_spam" 1 "needs_reply" 1})) "only one can still be true")
      (is (nil? (ev (into ["at-least" 2] xs) {"is_spam" 1})))
      (is (false? (ev (into ["at-most" 1] xs) {"is_spam" 0 "category" 0})))
      (is (true? (ev (into ["at-most" 1] xs) {"is_spam" 1 "needs_reply" 1})) "two falses: at most one left")
      (is (true? (ev (into ["exactly" 0] xs) {"is_spam" 1 "needs_reply" 1 "category" 2}))))))

(deftest ordinal-nodes-read-the-level-domain
  (testing "undecided: the whole range is possible"
    (is (nil? (ev ["min-level" "urgency" 1] {})))
    (is (true? (ev ["min-level" "urgency" 0] {})) "every level is at least 0")
    (is (true? (ev ["max-level" "urgency" 2] {})))
    (is (nil? (ev ["between-level" "urgency" 1 2] {}))))
  (testing "decided"
    (is (true? (ev ["min-level" "urgency" 1] {"urgency" 2})))
    (is (false? (ev ["min-level" "urgency" 1] {"urgency" 0})))
    (is (true? (ev ["max-level" "urgency" 1] {"urgency" 1})))
    (is (false? (ev ["max-level" "urgency" 1] {"urgency" 2})))
    (is (true? (ev ["at-level" "urgency" 1] {"urgency" 1})))
    (is (false? (ev ["between-level" "urgency" 1 2] {"urgency" 0})))))

;; --- decoding -----------------------------------------------------------------------

(def probs
  {"category" [0.6 0.3 0.1]
   "urgency" [0.2 0.5 0.3]
   "is_spam" [0.7 0.3]
   "needs_reply" [0.8 0.2]})

(defn- decode [constraints & [opts]]
  (c/decode schema probs (c/parse schema constraints) (or opts {})))

(deftest without-constraints-the-decision-is-the-argmax
  (let [r (decode [])]
    (is (= {"category" 0 "urgency" 1 "is_spam" 0 "needs_reply" 0} (:assignment r)))
    (is (= "independent" (:decoder r)))
    (is (true? (:feasible r)))
    (is (= [] (:violations r)))
    (is (true? (:exact r)))))

(deftest a-single-question-constraint-picks-the-best-allowed-label
  (let [r (decode [["not" ["category" "billing"]]])]
    (is (= "independent" (:decoder r)) "no cross-question coupling")
    (is (= 1 (get (:assignment r) "category")) "the runner-up")
    (is (true? (:feasible r)))))

(deftest cross-question-constraints-maximise-the-joint-log-probability
  (testing "spam implies no reply: flipping is_spam (0.7 -> 0.3) costs less than flipping needs_reply (0.8 -> 0.2)"
    (let [r (decode [["implies" ["is_spam" true] ["needs_reply" false]]])]
      (is (= "exact" (:decoder r)))
      (is (= {"category" 0 "urgency" 1 "is_spam" 1 "needs_reply" 0} (:assignment r)))
      (is (true? (:feasible r)))
      (is (true? (:exact r)))
      (is (< (Math/abs (- (:score r) (+ (Math/log 0.6) (Math/log 0.5) (Math/log 0.3) (Math/log 0.8)))) 1e-9))))
  (testing "iff pulls the cheaper side"
    (let [r (decode [["iff" ["is_spam" true] ["urgency" 2]]])]
      ;; spam & urgent: log .7 + log .3 = -1.56; not spam & not urgent (best level 1): log .3 + log .5 = -1.90
      (is (= {"category" 0 "urgency" 2 "is_spam" 0 "needs_reply" 0} (:assignment r)))))
  (testing "counting across questions"
    (let [r (decode [["at-most" 1 ["is_spam" true] ["needs_reply" true] ["category" "billing"]]])]
      ;; keep the most expensive to flip: needs_reply (0.8); flip is_spam (0.7->0.3) and category (0.6->0.3)
      (is (= {"category" 1 "urgency" 1 "is_spam" 1 "needs_reply" 0} (:assignment r))))))

(deftest infeasible-constraints-fall-to-the-fewest-violations
  (let [r (decode [["all-of" ["is_spam" true] ["is_spam" false]]
                   ["implies" ["is_spam" true] ["needs_reply" false]]])]
    (is (= "min_violations" (:decoder r)))
    (is (false? (:feasible r)))
    (is (= [0] (:violations r)) "only the contradiction is violated; the implication is honoured")
    (is (= [1 0] [(get (:assignment r) "is_spam") (get (:assignment r) "needs_reply")]))
    (is (true? (:exact r)) "the search finished, so the violation set is minimal"))
  (testing "or raise"
    (let [e (try (decode [["all-of" ["is_spam" true] ["is_spam" false]]] {:on-infeasible :raise})
                 nil (catch Exception e e))]
      (is (= :infeasible (:type (ex-data e))))
      (is (= [0] (:violations (ex-data e)))))))

(deftest a-blown-node-budget-falls-to-beam-search
  (let [r (decode [["implies" ["is_spam" true] ["needs_reply" false]]] {:budget 2})]
    (is (= "beam" (:decoder r)))
    (is (false? (:exact r)))
    (is (true? (:feasible r)))
    (is (= {"category" 0 "urgency" 1 "is_spam" 1 "needs_reply" 0} (:assignment r))
        "a beam of 16 over 4 questions still finds the optimum")))

;; --- exact decoder vs brute force on random problems -------------------------------

(defn- lcg
  "rand_r's recurrence; a lazy stream of doubles in [0, 1)."
  [seed]
  (rest (iterate (fn [x] (mod (+ (* x 1103515245) 12345) 2147483648)) seed)))

(defn- random-problem
  "3-5 questions of 2-4 labels with random calibrated probabilities and 1-3
  random constraints over random refs."
  [seed]
  (let [rs (atom (lcg seed))
        next! (fn [] (let [x (first @rs)] (swap! rs rest) (/ (double x) 2147483648.0)))
        int! (fn [n] (long (Math/floor (* n (next!)))))
        nq (+ 3 (int! 3))
        qids (mapv #(str "q" %) (range nq))
        schema (into {} (map (fn [q] (let [k (+ 2 (int! 3))]
                                       [q {:type "choice" :labels (mapv #(str "l" %) (range k))}]))
                             qids))
        probs (into {} (map (fn [q] (let [raw (repeatedly (count (get-in schema [q :labels])) #(+ 0.05 (next!)))
                                          s (reduce + raw)]
                                      [q (mapv #(/ % s) raw)]))
                            qids))
        ref! (fn [] (let [q (nth qids (int! nq))]
                      [q (nth (get-in schema [q :labels]) (int! (count (get-in schema [q :labels]))))]))
        node! (fn node! [depth]
                (let [k (int! (if (pos? depth) 7 4))]
                  (case k
                    0 (ref!)
                    1 ["not" (ref!)]
                    2 ["implies" (ref!) (ref!)]
                    3 ["excludes" (ref!) (ref!)]
                    4 ["iff" (node! (dec depth)) (ref!)]
                    5 (into ["at-most" (int! 2)] (repeatedly (+ 2 (int! 3)) ref!))
                    6 (into ["exactly-one-of"] (repeatedly (+ 2 (int! 3)) ref!)))))
        constraints (vec (repeatedly (+ 1 (int! 3)) #(node! 1)))]
    {:schema schema :probs probs :constraints constraints}))

(defn- brute-force
  "Every full assignment, scored: the best feasible one, else the fewest
  violations then the best score."
  [schema probs nodes]
  (let [qids (vec (sort (keys schema)))
        all (reduce (fn [acc q] (for [a acc i (range (count (get-in schema [q :labels])))] (assoc a q i)))
                    [{}] qids)
        scored (map (fn [a] {:assignment a
                             :score (reduce + (map (fn [q] (Math/log (nth (probs q) (a q)))) qids))
                             :violations (count (filter #(false? (c/evaluate % schema a)) nodes))})
                    all)
        best (apply min-key (fn [{:keys [violations score]}] (+ (* 1000.0 violations) (- score))) scored)]
    best))

(deftest exact-decoder-agrees-with-brute-force
  (doseq [seed (range 1 61)]
    (let [{:keys [schema probs constraints]} (random-problem seed)
          nodes (c/parse schema constraints)
          got (c/decode schema probs nodes {})
          want (brute-force schema probs nodes)]
      (is (= (count (:violations got)) (:violations want)) (str "seed " seed ": violation count " constraints))
      (is (< (Math/abs (- (:score got) (:score want))) 1e-9)
          (str "seed " seed ": score " constraints " got " (:assignment got) " want " (:assignment want)))
      (is (true? (:exact got)) (str "seed " seed))
      (when (:feasible got)
        (is (contains? #{"exact" "independent"} (:decoder got)) (str "seed " seed))))))
