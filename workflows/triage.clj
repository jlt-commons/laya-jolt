(ns workflows.triage
  "Support ticket triage (the upstream presets.triage_questions): intent,
  urgency, frustration, refund, churn. State: {\"message\" text}; a bare
  string is taken as the message.")

(defn questions
  "Support ticket triage: intent, urgency, frustration, refund requested, churn risk."
  []
  (array-map
   "intent" (array-map
             "type" "choice"
             "instructions" "What does the customer want in `message`?"
             "criteria" (array-map
                         "refund" "money returned or a duplicate charge reversed"
                         "technical_help" "a bug, outage or integration problem"
                         "billing_question" "a question about an invoice, plan or payment method"
                         "information" "general information, pricing or how-to"
                         "cancellation" "wants to cancel or downgrade"
                         "other" "none of the other options fits"))
   "is_urgent" (array-map
                "type" "noul"
                "instructions" "Does `message` communicate time pressure or a deadline?")
   "frustration" (array-map
                  "type" "score"
                  "instructions" "How frustrated does the customer sound in `message`?"
                  "criteria" ["calm and neutral"
                              "concerned but civil"
                              "clearly annoyed"
                              "very angry or using strong language"])
   "refund_requested" (array-map
                       "type" "noul"
                       "instructions" "Does the customer ask for money back?")
   "churn_risk" (array-map
                 "type" "noul"
                 "instructions" "Does `message` suggest the customer may leave for a competitor or cancel?")))

(defn state [input]
  (if (map? input) input {"message" (str input)}))
