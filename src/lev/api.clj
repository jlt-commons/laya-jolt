(ns lev.api
  "The library face von's users know (von.api / von.types): question
  constructors and one-question conveniences over system-one.

  `agent` is any agent (lev.agent/load-agent, lev.think/thinker) or a
  lev.router router, in which case the request is routed like an HTTP
  one (:model / :lang / :task in opts) and the answer carries \"routing\"."
  (:require [lev.agent :as ag]
            [lev.router :as router]
            [lev.sequence :as seq]))

;; --- questions ---------------------------------------------------------------

(defn choice
  "A choice question: criteria as {option description} (an ordered map:
  option order is model input) or a list of options."
  [instructions criteria]
  (array-map "type" "choice" "instructions" instructions
             "criteria" (if (map? criteria) criteria (seq/ordered-map (map (fn [c] [c nil]) criteria)))))

(defn noul
  "A yes/no question, optionally with {\"true\" .. \"false\" ..} criteria."
  ([instructions] (array-map "type" "noul" "instructions" instructions))
  ([instructions criteria] (array-map "type" "noul" "instructions" instructions "criteria" criteria)))

(defn score
  "An ordinal question over the levels in `criteria` (2 to 10 strings)."
  [instructions criteria]
  (array-map "type" "score" "instructions" instructions "criteria" (vec criteria)))

;; --- calls ---------------------------------------------------------------------

(defn system-one
  "state + questions on an agent or a router. opts: those of
  lev.agent/system-one (:constraints :on-infeasible :thinking :thought)
  and, on a router, :model :lang :task."
  ([agent state questions] (system-one agent state questions nil))
  ([agent state questions {:keys [model lang task] :as opts}]
   (if (router/router? agent)
     (apply router/predict agent state questions
            (mapcat identity (merge (dissoc opts :model :lang :task)
                                    {:model model :lang lang :task task})))
     (ag/system-one agent state questions (dissoc opts :model :lang :task)))))

(defn decide
  "One choice: `choices` a list of options or {option description};
  answers the choice answer map (choice, probabilities, confidence, ...)."
  ([agent state choices] (decide agent state choices "Which option best describes the state?" nil))
  ([agent state choices instructions] (decide agent state choices instructions nil))
  ([agent state choices instructions opts]
   (get-in (system-one agent state {"decision" (choice instructions choices)} opts) ["answers" "decision"])))

(defn judge
  "One yes/no: the probability that it holds."
  ([agent state instructions] (judge agent state instructions nil nil))
  ([agent state instructions criteria] (judge agent state instructions criteria nil))
  ([agent state instructions criteria opts]
   (get-in (system-one agent state {"judgment" (if criteria (noul instructions criteria) (noul instructions))} opts)
           ["answers" "judgment" "noul"])))

(defn rate
  "One ordinal rating over `criteria`; answers the score answer map."
  ([agent state criteria] (rate agent state criteria "Rate where the state falls on this scale:" nil))
  ([agent state criteria instructions] (rate agent state criteria instructions nil))
  ([agent state criteria instructions opts]
   (get-in (system-one agent state {"rating" (score instructions criteria)} opts) ["answers" "rating"])))
