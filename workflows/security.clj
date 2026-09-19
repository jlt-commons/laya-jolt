(ns workflows.security
  "Security event triage (von's presets.security_preset, ported): the kind
  of anomaly, whether it is an active threat, its severity. State:
  {\"event\" text} (a log line, an alert, a description); a bare string is
  taken as the event. Constraints tie the three together: a benign event
  is not a threat, and an active threat is at least elevated.")

(defn questions
  "Security event triage: event type, active threat, severity."
  []
  (array-map
   "event_type" (array-map
                 "type" "choice"
                 "instructions" "Classify the security or authentication anomaly in `event`."
                 "criteria" (array-map
                             "benign" "expected user activity, a legitimate IP change, a normal login"
                             "credential_stuffing" "a rapid succession of failed logins across many accounts"
                             "brute_force" "repeated failed attempts against one high-value account"
                             "privilege_escalation" "attempts at unauthorized administrative or sudo operations"
                             "data_exfiltration" "an abnormal volume of export requests or bulk downloads"))
   "is_threat" (array-map
                "type" "noul"
                "instructions" "Does `event` represent an active, confirmed malicious security threat?"
                "criteria" (array-map "true" "an active attack, intrusion or unauthorized compromise"
                                      "false" "an operational glitch, user error or benign variance"))
   "severity" (array-map
               "type" "score"
               "instructions" "Rate the severity of the incident in `event`."
               "criteria" ["informational: logged for the audit trail, no action"
                           "warning: suspicious variance, a rate limit triggered"
                           "elevated: an incident responder is paged"
                           "critical: an active breach; revoke tokens and ban the IP now"])))

(defn state [input]
  (if (map? input) input {"event" (str input)}))

(defn constraints
  "benign is never a threat; an active threat is at least elevated."
  []
  [[:implies ["event_type" "benign"] ["is_threat" false]]
   [:implies ["is_threat" true] [:min-level "severity" 2]]])
