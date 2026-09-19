(ns lev.agent
  "Agent.system_one: state + typed questions -> calibrated typed answers.
  Mirrors laya 0.3.0 agent.py (temperature calibration, entropy confidence,
  the `action` extension)."
  (:require [clojure.edn :as edn]
            [lev.constraints :as c]
            [lev.model :as m]
            [lev.sequence :as seq]
            [lev.tokenizer :as tk]))

(defn with-limits
  "The agent with :max-len / :head-max-len overridden (only the keys given).
  The checkpoint's trained max_len is kept as :trained-max-len for the logs:
  RoPE has no position table, so a longer sequence works mechanically, but
  quality past the trained length is unmeasured. Throws on values that
  cannot form a sequence."
  [agent {:keys [max-len head-max-len] :as limits}]
  (let [cfg (:cfg agent)
        max-len (or max-len (:max-len cfg))
        head-max-len (or head-max-len (:head-max-len cfg))
        bad (fn [msg] (throw (ex-info (str "invalid sequence limits " (pr-str limits) ": " msg)
                                      {:type :invalid-limits :limits limits})))]
    (when-not (and (integer? max-len) (integer? head-max-len)) (bad "max-len and head-max-len must be integers"))
    (when (< head-max-len 16) (bad "head-max-len must be at least 16 (build_sequence's own floor)"))
    (when (< max-len (+ head-max-len 8)) (bad "max-len must exceed head-max-len with room for the state"))
    (-> agent
        (update :cfg assoc :max-len max-len :head-max-len head-max-len)
        (assoc :trained-max-len (or (:trained-max-len agent) (:max-len cfg))))))

(defn load-agent
  "Load config + tokenizer + all weights from data-dir; `limits`
  ({:max-len :head-max-len}, see with-limits) override the checkpoint's."
  ([data-dir] (load-agent data-dir nil))
  ([data-dir limits]
   (let [manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
         cfg (edn/read-string (slurp (str data-dir "/config.edn")))
         tok (tk/load (str data-dir "/tokenizer.edn"))
         agent {:cfg cfg :tok tok :w (m/load-weights manifest data-dir) :trained-max-len (:max-len cfg)}]
     (if (seq limits) (with-limits agent limits) agent))))

(defn- qget
  "Question defs may carry keyword keys (Clojure literals) or string keys
  (parsed JSON, as the Python API receives them)."
  [qdef k]
  (if (contains? qdef k) (get qdef k) (get qdef (name k))))

(defn- invalid [qid field msg]
  (throw (ex-info (str "question " (pr-str qid) ": " msg)
                  {:type :invalid-question :qid qid :field field})))

(defn validate-question
  "The shape rl_agent_api needs (it fails with KeyError/AttributeError
  otherwise) and the Jev API documents: a known type, instructions, and
  criteria fitting the type. Throws ex-info {:type :invalid-question}."
  [qid qdef]
  (when-not (map? qdef) (invalid qid nil "must be an object"))
  (let [t (qget qdef :type)
        crit (qget qdef :criteria)]
    (when-not (contains? seq/qtypes t)
      (invalid qid "type" (str "unknown type " (pr-str t) "; expected choice, score or noul")))
    (when (nil? (qget qdef :instructions))
      (invalid qid "instructions" "instructions is required"))
    (case t
      "choice" (when-not (or (and (map? crit) (seq crit)) (and (sequential? crit) (seq crit)))
                 (invalid qid "criteria" "choice criteria must be a non-empty map of option -> description (or a list of options)"))
      "score" (when-not (and (sequential? crit) (>= (count crit) 2))
                (invalid qid "criteria" "score criteria must be a list of at least 2 levels"))
      "noul" (when-not (or (nil? crit) (map? crit))
               (invalid qid "criteria" "noul criteria must be a map with optional \"true\" / \"false\" descriptions")))
    qdef))

(defn to-internal
  "Jev question def -> {:t :ins :crit} (rl_agent_api.RLAgent._to_internal).
  A list of choice criteria becomes {c: None}; non-string instructions are
  json.dumps'd with its default ensure_ascii=True."
  [qdef]
  (let [t (qget qdef :type)
        crit (qget qdef :criteria)
        crit (if (and (= t "choice") (sequential? crit))
               (seq/ordered-map (map (fn [c] [c nil]) crit))
               crit)]
    {:t t
     :ins (let [ins (qget qdef :instructions)]
            (if (string? ins) ins (seq/json-str ins {:ensure-ascii true})))
     :crit crit}))

(def max-batch
  "Rows per forward-batch. Padded to the longest row, 8 rows of 512 tokens
  take a ~260 MB workspace; more rows buy little once the gemms are wide."
  8)

(defn- round4 [x] (/ (double (Math/round (* 1e4 (double x)))) 1e4))

(defn- softmax [xs]
  (let [mx (reduce max xs)
        ex (mapv #(Math/exp (- (double %) mx)) xs)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn confidence-from-probs
  "1 - normalized entropy, clipped to [0, 1] (confidence_from_probs)."
  [p k]
  (if (< k 2)
    1.0
    (let [ent (- (reduce + (map (fn [pi] (* pi (Math/log (max pi 1e-12)))) (take k p))))]
      (-> (- 1.0 (/ ent (Math/log k))) (max 0.0) (min 1.0)))))

(defn- argmax [xs]
  (reduce (fn [bi i] (if (> (nth xs i) (nth xs bi)) i bi)) 0 (range (count xs))))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- answer-for
  "One typed answer. With a decision (`decided`, the label the constrained
  decoder settled on) it follows the model's own answer field; without,
  the map is laya 0.3.0's exactly."
  [q p k actp decided]
  (let [ext (array-map "act_probability" (round4 actp))
        decided-kv (when (some? decided) [["decided" decided]])]
    (seq/ordered-map
     (case (:t q)
       "choice"
       (let [ks (mapv key-str (keys (:crit q)))]
         (concat [["type" "choice"] ["choice" (nth ks (argmax p))]]
                 decided-kv
                 [["probabilities" (seq/ordered-map (map-indexed (fn [i c] [c (round4 (nth p i))]) ks))]
                  ["confidence" (round4 (confidence-from-probs p k))]
                  ["action" ext]]))
       "score"
       (concat [["type" "score"] ["score" (round4 (reduce + (map-indexed (fn [i pi] (* i pi)) p)))]]
               decided-kv
               [["legend" (seq/ordered-map (map-indexed (fn [i c] [(str i) c]) (:crit q)))]
                ["probabilities" (seq/ordered-map (map-indexed (fn [i pi] [(str i) (round4 pi)]) p))]
                ["confidence" (round4 (confidence-from-probs p k))]
                ["action" ext]])
       (let [p1 (double (nth p 1))]
         (concat [["type" "noul"] ["noul" (round4 p1)]]
                 decided-kv
                 [["confidence" (round4 (max p1 (- 1.0 p1)))]
                  ["action" ext]]))))))

(defn- label-probs
  "p in the constraint schema's label order: a noul's options are rendered
  false first (\"noul\" is p[1]), its constraint labels are [true false]."
  [q p]
  (if (= "noul" (:t q)) [(nth p 1) (nth p 0)] p))

(defn- constraints-report [nodes sol]
  (seq/ordered-map [["feasible" (:feasible sol)]
                    ["decoder" (:decoder sol)]
                    ["exact" (:exact sol)]
                    ["violations" (mapv #(c/canonical (nth nodes %)) (:violations sol))]]))

(defn system-one
  "state + {qid -> qdef} -> Jev answer map (ordered to match json.dumps).

  opts (all optional): :constraints, a list of lev.constraints over the
  question ids, decided jointly after the forward pass. When it is given
  (even empty) every answer carries `decided` (choice: the option; score:
  the level index; noul: the boolean) after its own answer field, and the
  result a `constraints` report {feasible decoder exact violations}; the
  model's own fields never change. :on-infeasible `min_violations`
  (default: the fewest violated constraints, then the best score) or
  `raise` (ex-info {:type :infeasible :violations [...]}). A bad
  constraint is ex-info {:type :invalid-constraint :index i}, thrown
  before any inference."
  ([agent state questions] (system-one agent state questions nil))
  ([agent state questions {:keys [constraints on-infeasible]}]
  (let [{:keys [cfg tok w]} agent
        qids (vec (keys questions))
        prepared (mapv (fn [qid]
                         (let [qdef (validate-question qid (get questions qid))
                               q (to-internal qdef)
                               [ids markers] (seq/build-sequence tok state q
                                                                 (:max-len cfg)
                                                                 (:head-max-len cfg))]
                           (when (not= (count markers) (count (seq/render-options q)))
                             (throw (ex-info (format "question %s: options do not fit in head_max_len=%d tokens"
                                                     (pr-str qid) (:head-max-len cfg))
                                             {:type :invalid-question :qid qid :field "criteria"
                                              :head-max-len (:head-max-len cfg)})))
                           {:qid qid :qdef qdef :q q :ids ids :markers markers :qtype (seq/qtypes (:t q))}))
                       qids)
        ;; the constraints are checked against the questions before the
        ;; forward pass, so a bad one costs nothing
        schema (when (some? constraints)
                 (c/schema (map (fn [{:keys [qid qdef]}] [(key-str qid) qdef]) prepared)))
        nodes (when schema (c/parse schema constraints))
        n-tokens (reduce + (map #(count (:ids %)) prepared))
        ;; the questions share every gemm of the forward, in batches of at
        ;; most max-batch rows so the workspace stays a few hundred MB
        outputs (mapcat (fn [chunk]
                          (m/forward-batch w cfg
                                           (mapv (fn [{:keys [ids markers qtype]}]
                                                   {:ids ids :att (vec (repeat (count ids) 1))
                                                    :markers markers :marker-mask (vec (repeat (count markers) 1))
                                                    :qtype qtype})
                                                 chunk)))
                        (partition-all max-batch prepared))
        calibrated (mapv (fn [{:keys [qtype markers]} [logits act]]
                           (let [k (count markers)
                                 ;; max(1e-3, t_scale): a degenerate fitted temperature
                                 ;; saturates the softmax instead of dividing by zero
                                 temp (max 1e-3 (double (get (:temperature-by-options cfg)
                                                             (seq/temp-bucket qtype k)
                                                             (nth (:temperature cfg) qtype))))]
                             {:k k
                              :p (softmax (mapv #(/ (double %) temp) (take k logits)))
                              :actp (first (softmax act))}))
                         prepared outputs)
        solution (when nodes
                   (let [sol (c/decode schema
                                       (into {} (map (fn [{:keys [qid q]} {:keys [p]}] [(key-str qid) (label-probs q p)])
                                                     prepared calibrated))
                                       nodes {})]
                     (when (and (not (:feasible sol)) (= "raise" (some-> on-infeasible name)))
                       (throw (ex-info "no assignment satisfies the constraints"
                                       {:type :infeasible
                                        :violations (mapv #(c/canonical (nth nodes %)) (:violations sol))})))
                     sol))
        decided-label (fn [qid]
                        (when solution
                          (let [q (key-str qid)]
                            (nth (get-in schema [q :labels]) (get-in solution [:assignment q])))))
        answers (seq/ordered-map
                 (map (fn [{:keys [qid q]} {:keys [k p actp]}]
                        [qid (answer-for q p k actp (decided-label qid))])
                      prepared calibrated))]
    (seq/ordered-map
     (concat [["model" "laya-rl-agent"]
              ["answers" answers]
              ["usage" (array-map "input_tokens" n-tokens "output_tokens" 0)]]
             (when solution [["constraints" (constraints-report nodes solution)]]))))))
