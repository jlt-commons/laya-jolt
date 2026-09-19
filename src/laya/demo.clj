(ns laya.demo
  "README quickstart: run the full stack on the example email and print the
  JSON answer. Mirror of the Python `RLAgent.system_one` example.

  Usage: jolt -M:run [data-dir]   (default data-dir: \"data\")"
  (:require [laya.agent :as ag]
            [laya.sequence :as seq]))

(defn run
  "state + questions -> answer map."
  [agent]
  (let [state (array-map
               "from" "customer@acme.com"
               "subject" "Duplicate billing on March invoice #4411"
               "body" (str "Hi team, we were billed twice for March. Please refund "
                           "the duplicate before Friday or we will cancel our plan."))
        questions (array-map
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
                                  :instructions "Is this email a phishing or scam attempt?"})]
    (ag/system-one agent state questions)))

(defn -main
  [& [data-dir]]
  (let [agent (ag/load-agent (or data-dir "data"))]
    (println (seq/json-str (run agent)))))
