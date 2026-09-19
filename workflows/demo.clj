(ns workflows.demo
  "README quickstart as a workflow: the example email and its four questions,
  mirroring the Python `Agent.system_one` example. Its answer must match
  golden/readme.edn :system-one to the fourth decimal.

  Usage: jolt -M:run demo")

(def example-state
  (array-map
   "from" "customer@acme.com"
   "subject" "Duplicate billing on March invoice #4411"
   "body" (str "Hi team, we were billed twice for March. Please refund "
               "the duplicate before Friday or we will cancel our plan.")))

(defn questions
  "The README quickstart: department, urgency, churn risk, phishing."
  []
  (array-map
   "department" {:type "choice"
                 :instructions "Which department should handle this email?"
                 :criteria (array-map
                            "billing" "invoices, payments, refunds"
                            "technical" "bugs, outages, integrations"
                            "sales" "pricing, contracts, demos"
                            "other" "everything else")}
   "urgency" {:type "score"
              :instructions "How urgent is this request?"
              :criteria ["not urgent" "soon"
                         "critical deadline or blocking issue"]}
   "churn_risk" {:type "noul"
                 :instructions (str "Does the user threaten to cancel "
                                    "or switch to a competitor?")}
   "is_phishing" {:type "noul"
                  :instructions "Is this email a phishing or scam attempt?"}))

(defn state
  "The quickstart email, whatever the input (nil or {} pick the example)."
  [input]
  (if (and (map? input) (seq input)) input example-state))
