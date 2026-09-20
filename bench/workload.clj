(ns bench.workload
  "The multi-question call bench/workflow.clj and bench/paired.clj time:
  the README's four email questions (choice, score, noul, noul) over a
  short state (the README's email, 53 tokens) and a long one (a 1.6k-token
  email, cut to max_len for every question). `variants` gives n states
  that differ in their first words, for a benchmark that must not run the
  same input twice in a row. Loaded by the scripts with load-file, from
  the project root.")

(def questions
  (array-map
   "department" {"type" "choice"
                 "instructions" "Which department should handle this email?"
                 "criteria" (array-map "billing" "invoices, payments, refunds"
                                       "technical" "bugs, outages, integrations"
                                       "sales" "pricing, contracts, demos"
                                       "other" "everything else")}
   "urgency" {"type" "score"
              "instructions" "How urgent is this request?"
              "criteria" ["not urgent" "soon" "critical deadline or blocking issue"]}
   "churn_risk" {"type" "noul"
                 "instructions" "Does the user threaten to cancel or switch to a competitor?"}
   "is_phishing" {"type" "noul"
                  "instructions" "Is this email a phishing or scam attempt?"}))

(def short-state
  (array-map
   "from" "customer@acme.com"
   "subject" "Duplicate billing on March invoice #4411"
   "body" "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan."))

(def long-state
  (assoc short-state "body"
         (apply str (repeat 40 "We were billed twice for the same invoice INV-2291 last Tuesday and the support line was closed; please refund the duplicate charge to the card ending 4412 before Friday, or we cancel. "))))

(defn variants
  "n copies of state whose body starts \"Record i: \" (as laya-mlx's paired
  run varies its inputs): different tokens, the same length class."
  [state n]
  (mapv (fn [i] (update state "body" #(str "Record " i ": " %))) (range n)))

(def cases
  "name -> [state questions]"
  (array-map
   "short4" [short-state questions]
   "long4" [long-state questions]
   "long1" [long-state (select-keys questions ["department"])]))
