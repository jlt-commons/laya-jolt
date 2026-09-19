(ns workflows.moderation
  "Content safety and moderation (laya 0.3.0 presets.moderation_questions):
  toxicity, harassment, threats, spam, severity. State: {\"post\" text}; a
  bare string is taken as the post.")

(defn questions
  "Content moderation: toxic, harassment, threat, spam, severity."
  []
  (array-map
   "toxic" (array-map
            "type" "noul"
            "instructions" "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?")
   "harassment" (array-map
                 "type" "noul"
                 "instructions" "Does `post` target or harass a specific person?")
   "threat" (array-map
             "type" "noul"
             "instructions" "Does `post` threaten violence, harm or intimidation?")
   "spam" (array-map
           "type" "noul"
           "instructions" "Is `post` spam or advertising?")
   "severity" (array-map
               "type" "score"
               "instructions" "How severe is any rule-breaking in `post`?"
               "criteria" ["no rule-breaking: ordinary on-topic post"
                           "mild: rude tone or off-topic, no target"
                           "clear violation: insults, harassment or spam aimed at someone"
                           "severe: threats, hate speech or calls for violence"])))

(defn state [input]
  (if (map? input) input {"post" (str input)}))
