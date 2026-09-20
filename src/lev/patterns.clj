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

(def default-threshold 0.8)

(defn thresholds
  "The threshold per question type from a number (every type) or a map
  {type threshold} (a type left out gets the default 0.8): {\"choice\" t
  \"score\" t \"noul\" t}. Throws {:type :invalid-request} on anything else."
  [t]
  (let [bad (fn [what] (throw (ex-info (str "threshold must be a number in [0, 1] or {type: number}; " what)
                                       {:type :invalid-request :field "threshold"})))
        ok? (fn [x] (and (number? x) (<= 0.0 (double x) 1.0)))]
    (cond
      (nil? t) (thresholds default-threshold)
      (number? t) (if (ok? t) (zipmap ["choice" "score" "noul"] (repeat (double t))) (bad (str "got " (pr-str t))))
      (map? t) (do (doseq [[k v] t]
                     (when-not (contains? #{"choice" "score" "noul"} (key-str k)) (bad (str "unknown type " (pr-str k))))
                     (when-not (ok? v) (bad (str "got " (pr-str v) " for " (pr-str k)))))
                   (into {} (map (fn [type] [type (double (get t type (get t (keyword type) default-threshold)))])
                                 ["choice" "score" "noul"])))
      :else (bad (str "got " (pr-str t))))))

(defn- confident? [answer ths]
  (>= (double (get answer "confidence" 1.0)) (get ths (get answer "type") default-threshold)))

(defn confidence-gate
  "Selective automation: the answers at or above :threshold (0.8; a
  number, or {type threshold} — see `thresholds`) under \"automatic\",
  the rest under \"escalate\", the whole response under \"response\".
  Other opts go to lev.api/system-one."
  [agent state questions {:keys [threshold] :as opts}]
  (let [ths (thresholds threshold)
        resp (api/system-one agent state questions (dissoc opts :threshold))
        [auto esc] (reduce (fn [[a e] [qid answer]]
                             (if (confident? answer ths)
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
  [rt state questions {:keys [threshold model fast-model lang task thinking thought debias constraints on-infeasible]}]
  (when-not (router/router? rt)
    (throw (ex-info "escalation needs a router (the thinker is named by model)" {:type :invalid-request})))
  (let [ths (thresholds threshold)
        model (router/normalise-name rt (or model (throw (ex-info "escalate: :model (the thinker) is required"
                                                                 {:type :invalid-request :field "model"}))))
        fast (router/predict rt state questions :model fast-model :lang lang :task task :debias debias)
        unsure (vec (keep (fn [[qid a]] (when-not (confident? a ths) qid)) (get fast "answers")))
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
              ["escalation" (seq/ordered-map [["threshold" (if (and (number? threshold) (apply = (vals ths))) (double threshold) (seq/ordered-map (map (fn [t] [t (get ths t)]) ["choice" "score" "noul"])))]
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
