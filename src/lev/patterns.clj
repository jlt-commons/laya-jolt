(ns lev.patterns
  "Composable decision patterns over system-one (von.patterns, ported):
  a confidence gate, route dispatch, a composite risk score, two-stage
  routing for wide taxonomies — and escalation, the pattern the
  authored144 comparison (bench/) argues for: answer with the fast
  encoder, and re-ask only the questions it is unsure about on a thinker
  (lev.think), seconds a question but right where the encoder is not.

  `agent` is an agent or a lev.router router (lev.api/system-one);
  escalation needs the router, since it names the thinker by model name.
  Confidence is each answer's own \"confidence\": for choice and score
  1 - normalised entropy, for noul max(p, 1 - p)."
  (:require [lev.agent :as ag]
            [lev.api :as api]
            [lev.router :as router]
            [lev.sequence :as seq]))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- check-threshold [t]
  (when-not (and (number? t) (<= 0.0 (double t) 1.0))
    (throw (ex-info (str "threshold must be in [0, 1]; got " (pr-str t)) {:type :invalid-request :field "threshold"})))
  (double t))

(defn- confident? [answer threshold]
  (>= (double (get answer "confidence" 1.0)) threshold))

(defn confidence-gate
  "Selective automation: the answers at or above :threshold (0.8) under
  \"automatic\", the rest under \"escalate\", the whole response under
  \"response\". Other opts go to lev.api/system-one."
  [agent state questions {:keys [threshold] :or {threshold 0.8} :as opts}]
  (let [threshold (check-threshold threshold)
        resp (api/system-one agent state questions (dissoc opts :threshold))
        [auto esc] (reduce (fn [[a e] [qid answer]]
                             (if (confident? answer threshold)
                               [(conj a [qid answer]) e]
                               [a (conj e [qid answer])]))
                           [[] []] (get resp "answers"))]
    (seq/ordered-map [["automatic" (seq/ordered-map auto)]
                      ["escalate" (seq/ordered-map esc)]
                      ["response" resp]])))

(defn escalate
  "Answer on the fast model (routed, or :fast-model), then re-ask the
  questions below :threshold (0.8) on :model (a thinker; :thinking /
  :thought as for it) and merge: the answers keep their order, the
  escalated ones are the thinker's, and an \"escalation\" report names
  them with the thinker's usage. :constraints / :on-infeasible decide
  over the merged answers. Needs a router."
  [rt state questions {:keys [threshold model fast-model lang task thinking thought constraints on-infeasible]
                       :or {threshold 0.8}}]
  (when-not (router/router? rt)
    (throw (ex-info "escalation needs a router (the thinker is named by model)" {:type :invalid-request})))
  (let [threshold (check-threshold threshold)
        model (router/normalise-name rt (or model (throw (ex-info "escalate: :model (the thinker) is required"
                                                                 {:type :invalid-request :field "model"}))))
        fast (router/predict rt state questions :model fast-model :lang lang :task task)
        unsure (vec (keep (fn [[qid a]] (when-not (confident? a threshold) qid)) (get fast "answers")))
        unsure-set (set unsure)
        slow (when (seq unsure)
               (router/predict rt state (seq/ordered-map (filter (fn [[qid _]] (contains? unsure-set (key-str qid))) questions))
                               :model model :thinking thinking :thought thought))
        merged (seq/ordered-map
                (map (fn [[qid a]] [qid (if (contains? unsure-set qid) (get-in slow ["answers" qid]) a)])
                     (get fast "answers")))
        cs (ag/prepare-constraints (map (fn [[qid qdef]] [qid (ag/validate-question qid qdef)]) questions) constraints)
        sol (when cs
              (ag/decide-constraints cs (map (fn [[qid qdef]] [qid (ag/to-internal qdef) (ag/answer-probs (get merged (key-str qid)))])
                                             questions)
                                     on-infeasible))
        answers (if sol
                  (seq/ordered-map (map (fn [[qid a]] [qid (ag/with-decided a (ag/decided-label cs sol qid))]) merged))
                  merged)]
    (seq/ordered-map
     (concat [["model" (get fast "model")]
              ["answers" answers]
              ["usage" (get fast "usage")]
              ["escalation" (seq/ordered-map [["threshold" threshold]
                                              ["model" model]
                                              ["escalated" unsure]
                                              ["usage" (or (get slow "usage") (array-map "input_tokens" 0 "output_tokens" 0))]])]]
             (when sol [["constraints" (ag/constraints-report cs sol)]])
             [["routing" (get fast "routing")]]))))

(defn route
  "Run one choice question and hand its answer to the handler in `routes`
  ({option (fn [answer])}) for the winning option. Without a handler, or
  with confidence below :min-confidence (0), :default gets the answer,
  or the answer itself is returned."
  [agent state question routes {:keys [default min-confidence] :or {min-confidence 0.0} :as opts}]
  (when-not (= "choice" (get question "type" (get question :type)))
    (throw (ex-info "route takes a choice question" {:type :invalid-request :field "question"})))
  (let [answer (get-in (api/system-one agent state {"route_question" question} (dissoc opts :default :min-confidence))
                       ["answers" "route_question"])
        handler (get routes (get answer "choice"))]
    (cond
      (and handler (>= (double (get answer "confidence")) (double min-confidence))) (handler answer)
      default (default answer)
      :else answer)))

(defn composite-score
  "One risk number in [0, 1] from the score and noul answers: a score is
  its expectation over the top level index, a noul its probability, each
  weighted by :weights {qid w} (1), averaged (or summed with :normalize
  false); choices are left out. Answers {\"score\" \"breakdown\" \"response\"}."
  [agent state questions {:keys [weights normalize] :or {normalize true} :as opts}]
  (let [resp (api/system-one agent state questions (dissoc opts :weights :normalize))
        rows (keep (fn [[qid a]]
                     (let [v (case (get a "type")
                               "score" (/ (double (get a "score")) (max 1 (dec (count (get a "legend")))))
                               "noul" (double (get a "noul"))
                               nil)]
                       (when v
                         (let [w (double (get weights qid 1.0))]
                           [qid (seq/ordered-map [["raw" (get a (get a "type"))] ["normalized" v] ["weight" w]])]))))
                   (get resp "answers"))
        total (reduce + 0.0 (map (fn [[_ r]] (* (get r "normalized") (get r "weight"))) rows))
        wsum (reduce + 0.0 (map (fn [[_ r]] (get r "weight")) rows))]
    (seq/ordered-map [["score" (if (and normalize (pos? wsum)) (/ total wsum) total)]
                      ["breakdown" (seq/ordered-map rows)]
                      ["response" resp]])))

(defn two-stage-choice
  "A wide taxonomy ({category {option description}}) in two choices: the
  category, then the option within it; combined confidence is the
  product."
  [agent state taxonomy {:keys [instructions-category instructions-option]
                         :or {instructions-category "Which broad category best matches the state?"
                              instructions-option "Which specific option applies within %s?"}
                         :as opts}]
  (let [opts (dissoc opts :instructions-category :instructions-option)
        cats (seq/ordered-map (map (fn [[c _]] [(key-str c) (str "Category for " (key-str c) " operations and topics")]) taxonomy))
        cat (api/decide agent state cats instructions-category opts)
        top (get cat "choice")
        subs (or (some (fn [[c os]] (when (= (key-str c) top) os)) taxonomy) {})
        opt (api/decide agent state subs (format instructions-option top) opts)]
    (seq/ordered-map [["category" top]
                      ["category_confidence" (get cat "confidence")]
                      ["choice" (get opt "choice")]
                      ["choice_confidence" (get opt "confidence")]
                      ["combined_confidence" (ag/round4 (* (double (get cat "confidence")) (double (get opt "confidence"))))]])))
