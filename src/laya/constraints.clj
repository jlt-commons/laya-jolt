(ns laya.constraints
  "Constrained decoding across a call's questions: the torch-free port of
  gliner2.classification (constraints.py, candidates.py, decoding/).

  system-one answers every question on its own; a constraint ties them.
  A constraint is data, the same in a request body, a workflow file or a
  test:

    [\"implies\" [\"is_spam\" true] [\"needs_reply\" false]]
    [\"at-most\" 1 [\"is_spam\" true] [\"needs_reply\" true]]
    [\"min-level\" \"urgency\" 1]

  A ref [question label] names one label of a question: a choice option,
  a score level (index or legend text) or a noul boolean. Operators (a
  string, keyword, dashes or underscores): not, all-of, any-of, implies,
  iff, excludes, exactly-one-of, at-least k, at-most k, exactly k (each
  over refs or nested constraints), and for score questions at-level,
  min-level, max-level, between-level. Every question here is exclusive
  (one label wins), so GLiNER2's per-task cardinality becomes counting
  over child expressions, and its default-label predicates have nothing
  to refer to.

  Evaluation is Kleene three-valued over a partial assignment ({question
  label-index}): true = satisfied, false = violated, nil = not yet
  decided; that is what lets the search prune before every question is
  decided. The decision maximises the sum of log probabilities (the joint
  under independence, so without constraints it is the argmax of each
  question): independent when nothing couples questions, else exact DFS
  with branch and bound on a node budget, beam search past the budget;
  an infeasible set falls to the fewest violations (then the best score)
  or raises, as the caller asks. A violating assignment is never reported
  as feasible."
  (:require [clojure.string :as str]))

;; --- the schema: what each question can decide on ---------------------------------

(defn- qget [qdef k] (if (contains? qdef k) (get qdef k) (get qdef (name k))))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn schema
  "{qid {:type :labels [..] :legend [..]}} from a validated question map
  ({qid qdef}, keyword or string keys as agent/validate-question takes):
  choice labels are the options in order, score labels the level indices
  (with the legend alongside), noul labels [true false]."
  [questions]
  (into {}
        (map (fn [[qid qdef]]
               (let [t (qget qdef :type)
                     crit (qget qdef :criteria)]
                 [(key-str qid)
                  (case t
                    "choice" {:type t :labels (mapv key-str (if (map? crit) (keys crit) crit))}
                    "score" {:type t :labels (vec (range (count crit))) :legend (vec crit)}
                    {:type t :labels [true false]})])))
        questions))

;; --- parsing --------------------------------------------------------------------

(def operators
  #{"not" "all-of" "any-of" "implies" "iff" "excludes" "exactly-one-of"
    "at-least" "at-most" "exactly" "at-level" "min-level" "max-level" "between-level"})

(defn- op-name
  "The operator a vector's head names, or nil: strings and keywords, any
  case, dashes or underscores."
  [x]
  (when (or (string? x) (keyword? x))
    (let [n (str/replace (str/lower-case (key-str x)) "_" "-")]
      (when (contains? operators n) n))))

(defn- index-of [v x] (first (keep-indexed (fn [i y] (when (= x y) i)) v)))

(defn- fail [index msg]
  (throw (ex-info (str "constraint " index ": " msg) {:type :invalid-constraint :index index})))

(defn- label-index
  "The index of `label` among question q's labels, resolving a score
  legend string and a noul \"true\"/\"false\"."
  [schema index q label]
  (let [spec (or (get schema q) (fail index (str "unknown question " (pr-str q))))
        labels (:labels spec)
        i (case (:type spec)
            "choice" (or (index-of labels label)
                         (fail index (str (pr-str label) " is not an option of " (pr-str q)
                                          "; options: " (str/join ", " (map pr-str labels)))))
            "score" (cond
                      (integer? label) (when (< -1 label (count labels)) label)
                      (string? label) (index-of (:legend spec) label))
            (cond
              (boolean? label) (if label 0 1)
              (= "true" label) 0
              (= "false" label) 1))]
    (or i
        (fail index (case (:type spec)
                      "score" (str (pr-str label) " is not a level of " (pr-str q) "; levels: 0.."
                                   (dec (count labels)) " or the legend text")
                      (str (pr-str label) " is not a noul value of " (pr-str q) "; use true or false"))))))

(defn- ref-node [schema index x]
  (when-not (and (vector? x) (= 2 (count x)) (or (string? (first x)) (keyword? (first x))))
    (fail index (str "a ref is [question label]; got " (pr-str x))))
  (let [q (key-str (first x))
        i (label-index schema index q (second x))]
    {:op :ref :q q :i i :label (nth (get-in schema [q :labels]) i)}))

(defn- ordinal [schema index op q level]
  (let [q (if (or (string? q) (keyword? q)) (key-str q) (fail index (str op " takes a question id; got " (pr-str q))))
        spec (or (get schema q) (fail index (str "unknown question " (pr-str q))))]
    (when-not (= "score" (:type spec))
      (fail index (str op " needs a score question; " (pr-str q) " is a " (:type spec))))
    (label-index schema index q level)))

(defn- count-arg [index op k]
  (when-not (and (integer? k) (not (neg? k)))
    (fail index (str op " takes a non-negative count first; got " (pr-str k))))
  k)

(defn- parse-node [schema index x]
  (cond
    (not (or (vector? x) (seq? x)))
    (fail index (str "a constraint is a vector [op ...] or a ref [question label]; got " (pr-str x)))

    (empty? x) (fail index "empty constraint")

    :else
    (let [x (vec x)
          op (op-name (first x))
          args (rest x)
          n (count args)
          sub (fn [y] (parse-node schema index y))
          subs (fn [ys] (mapv sub ys))
          arity (fn [want] (when-not (= want n) (fail index (str op " takes " want " argument" (when (not= 1 want) "s") "; got " n))))
          at-least-one (fn [ys] (when (empty? ys) (fail index (str op " needs at least one constraint"))))]
      (case op
        nil (ref-node schema index x)
        "not" (do (arity 1) {:op :not :x (sub (first args))})
        "all-of" (do (at-least-one args) {:op :and :xs (subs args)})
        "any-of" (do (at-least-one args) {:op :or :xs (subs args)})
        "implies" (do (arity 2) {:op :implies :a (sub (first args)) :b (sub (second args))})
        "iff" (do (arity 2) {:op :iff :a (sub (first args)) :b (sub (second args))})
        "excludes" (do (arity 2) {:op :excludes :a (sub (first args)) :b (sub (second args))})
        "exactly-one-of" (do (at-least-one args) {:op :count :min 1 :max 1 :xs (subs args)})
        "at-least" (let [k (count-arg index op (first args))] (at-least-one (rest args)) {:op :count :min k :max nil :xs (subs (rest args))})
        "at-most" (let [k (count-arg index op (first args))] (at-least-one (rest args)) {:op :count :min 0 :max k :xs (subs (rest args))})
        "exactly" (let [k (count-arg index op (first args))] (at-least-one (rest args)) {:op :count :min k :max k :xs (subs (rest args))})
        "at-level" (do (arity 2) (let [q (key-str (first args)) i (ordinal schema index op (first args) (second args))]
                                   {:op :ref :q q :i i :label i}))
        "min-level" (do (arity 2) {:op :min-level :q (key-str (first args)) :i (ordinal schema index op (first args) (second args))})
        "max-level" (do (arity 2) {:op :max-level :q (key-str (first args)) :i (ordinal schema index op (first args) (second args))})
        "between-level" (do (arity 3)
                            (let [q (key-str (first args))
                                  lo (ordinal schema index op (first args) (second args))
                                  hi (ordinal schema index op (first args) (nth args 2))]
                              (when (> lo hi) (fail index (str "between-level: " lo " is above " hi)))
                              {:op :and :xs [{:op :min-level :q q :i lo} {:op :max-level :q q :i hi}]}))))))

(defn parse
  "The constraint list (JSON or EDN data) as AST nodes, validated against
  the schema. Throws ex-info {:type :invalid-constraint :index i} naming
  the first bad constraint."
  [schema constraints]
  (when-not (sequential? constraints)
    (throw (ex-info (str "constraints must be a list of constraints; got " (pr-str constraints))
                    {:type :invalid-constraint})))
  (doseq [q (keys schema)]
    (when (op-name q)
      (throw (ex-info (str "question id " (pr-str q) " is a constraint operator; rename the question to use constraints")
                      {:type :invalid-constraint :qid q}))))
  (vec (map-indexed (fn [i x] (parse-node schema i x)) constraints)))

(defn canonical
  "A node back as data: operators as dashed strings, refs as [question
  label] with the label as the schema spells it (option, level index,
  boolean). Parses back to the same node."
  [node]
  (case (:op node)
    :ref [(:q node) (:label node)]
    :not ["not" (canonical (:x node))]
    :and (into ["all-of"] (map canonical (:xs node)))
    :or (into ["any-of"] (map canonical (:xs node)))
    :implies ["implies" (canonical (:a node)) (canonical (:b node))]
    :iff ["iff" (canonical (:a node)) (canonical (:b node))]
    :excludes ["excludes" (canonical (:a node)) (canonical (:b node))]
    :count (let [{:keys [min max xs]} node
                 xs (map canonical xs)]
             (cond
               (and (= 1 min) (= 1 max)) (into ["exactly-one-of"] xs)
               (= min max) (into ["exactly" min] xs)
               (nil? max) (into ["at-least" min] xs)
               :else (into ["at-most" max] xs)))
    :min-level ["min-level" (:q node) (:i node)]
    :max-level ["max-level" (:q node) (:i node)]))

(defn references
  "The question ids a node reads."
  [node]
  (case (:op node)
    (:ref :min-level :max-level) #{(:q node)}
    :not (references (:x node))
    (:and :or :count) (reduce into #{} (map references (:xs node)))
    (:implies :iff :excludes) (into (references (:a node)) (references (:b node)))))

;; --- Kleene evaluation -----------------------------------------------------------

(defn- k-not [v] (when (some? v) (not v)))

(defn- k-and [vs]
  (cond (some false? vs) false
        (some nil? vs) nil
        :else true))

(defn- k-or [vs]
  (cond (some true? vs) true
        (some nil? vs) nil
        :else false))

(defn- levels
  "The level indices question q can still take under `assign`."
  [schema assign q]
  (if (contains? assign q)
    [(get assign q)]
    (get-in schema [q :labels])))

(defn evaluate
  "true = satisfied, false = violated, nil = undetermined, over a partial
  assignment {qid label-index}."
  [node schema assign]
  (case (:op node)
    :ref (when (contains? assign (:q node)) (= (:i node) (get assign (:q node))))
    :not (k-not (evaluate (:x node) schema assign))
    :and (k-and (map #(evaluate % schema assign) (:xs node)))
    :or (k-or (map #(evaluate % schema assign) (:xs node)))
    :implies (k-or [(k-not (evaluate (:a node) schema assign)) (evaluate (:b node) schema assign)])
    :iff (let [a (evaluate (:a node) schema assign) b (evaluate (:b node) schema assign)]
           (when (and (some? a) (some? b)) (= a b)))
    :excludes (k-not (k-and [(evaluate (:a node) schema assign) (evaluate (:b node) schema assign)]))
    :count (let [vs (map #(evaluate % schema assign) (:xs node))
                 lo (count (filter true? vs))
                 hi (+ lo (count (filter nil? vs)))
                 mn (:min node)
                 mx (or (:max node) hi)]
             (cond (> lo mx) false
                   (< hi mn) false
                   (and (>= lo mn) (<= hi mx)) true
                   :else nil))
    :min-level (let [ls (levels schema assign (:q node)) floor (:i node)]
                 (cond (>= (reduce min ls) floor) true
                       (< (reduce max ls) floor) false
                       :else nil))
    :max-level (let [ls (levels schema assign (:q node)) ceil (:i node)]
                 (cond (<= (reduce max ls) ceil) true
                       (> (reduce min ls) ceil) false
                       :else nil))))

;; --- decoding ------------------------------------------------------------------

(defn- utilities
  "Per question, [[label-index log-p] ...] best first: the locals of
  candidates.py, one per label since every question is exclusive."
  [schema probs]
  (into {}
        (map (fn [[q spec]]
               [q (->> (map-indexed (fn [i p] [i (Math/log (max (double p) 1e-12))]) (take (count (:labels spec)) (get probs q)))
                       (sort-by (fn [[i u]] [(- u) i]))
                       vec)]))
        schema))

(defn- touching [nodes q]
  (vec (keep-indexed (fn [i n] (when (contains? (references n) q) i)) nodes)))

(defn- search-order
  "Most constrained first, then fewest labels, then the id: coupled
  questions decide early so the pruning bites."
  [schema nodes]
  (vec (sort-by (fn [q] [(- (count (touching nodes q))) (count (get-in schema [q :labels])) q])
                (keys schema))))

(defn- violations-of [nodes schema assign]
  (vec (keep-indexed (fn [i n] (when (false? (evaluate n schema assign)) i)) nodes)))

(defn- consistent?
  "No constraint touching q is violated by assign (q just decided)."
  [nodes schema assign touch]
  (every? (fn [i] (not (false? (evaluate (nth nodes i) schema assign)))) touch))

(defn- score-of [utils assign]
  (reduce + (map (fn [[q i]] (some (fn [[j u]] (when (= i j) u)) (utils q))) assign)))

(defn- result [decoder assign score violations exact?]
  {:assignment assign :score score :violations violations
   :feasible (empty? violations) :exact exact? :decoder decoder})

(defn- independent
  "Each question's best label that violates no constraint on it alone;
  base.py's IndependentDecoder."
  [schema nodes utils]
  (let [assign (into {}
                     (map (fn [q]
                            (let [touch (touching nodes q)
                                  locals (utils q)]
                              [q (or (some (fn [[i _]] (when (consistent? nodes schema {q i} touch) i)) locals)
                                     (ffirst locals))]))
                          (keys schema)))]
    (result "independent" assign (score-of utils assign) (violations-of nodes schema assign) true)))

(defn- exact
  "DFS with branch and bound; nil when nothing feasible, throws
  ::budget past `budget` nodes (exact.py's ExactDecoder)."
  [schema nodes utils budget]
  (let [order (search-order schema nodes)
        n (count order)
        ;; suffix[i]: the best the questions from i on can add, independently
        suffix (reduce (fn [s i] (assoc s i (+ (second (first (utils (nth order i)))) (nth s (inc i)))))
                       (vec (repeat (inc n) 0.0))
                       (range (dec n) -1 -1))
        best (atom {:assign nil :score ##-Inf})
        visited (atom 0)]
    (letfn [(dfs [i assign score]
              (when (> (swap! visited inc) budget) (throw (ex-info "budget" {:type ::budget})))
              (when (> (+ score (nth suffix i)) (:score @best))
                (if (= i n)
                  (reset! best {:assign assign :score score})
                  (let [q (nth order i)
                        touch (touching nodes q)]
                    (loop [locals (utils q)]
                      (when-let [[l u] (first locals)]
                        ;; locals are best first: past the bound, the rest are worse too
                        (when (> (+ score u (nth suffix (inc i))) (:score @best))
                          (let [a (assoc assign q l)]
                            (when (consistent? nodes schema a touch)
                              (dfs (inc i) a (+ score u))))
                          (recur (rest locals)))))))))]
      (dfs 0 {} 0.0)
      (when-let [a (:assign @best)]
        (result "exact" a (:score @best) [] true)))))

(defn- beam
  "Bounded search past the budget (beam.py); nil at a dead end rather
  than a violating answer dressed as feasible."
  [schema nodes utils width]
  (let [order (search-order schema nodes)]
    (loop [i 0 beams [[0.0 {}]]]
      (if (= i (count order))
        (let [[score assign] (first beams)]
          (result "beam" assign score (violations-of nodes schema assign) false))
        (let [q (nth order i)
              touch (touching nodes q)
              expanded (for [[score assign] beams
                             [l u] (utils q)
                             :let [a (assoc assign q l)]
                             :when (consistent? nodes schema a touch)]
                         [(+ score u) a])
              kept (->> expanded
                        (sort-by (fn [[s a]] [(- s) (mapv a (take (inc i) order))]))
                        (take width)
                        vec)]
          (when (seq kept) (recur (inc i) kept)))))))

(defn- min-violations
  "Every assignment, fewest violations then best score (exact.py's
  MinViolationsDecoder); :exact false when the budget cut the search."
  [schema nodes utils budget]
  (let [order (search-order schema nodes)
        n (count order)
        best (atom {:assign nil :weight Long/MAX_VALUE :score ##-Inf})
        visited (atom 0)
        exhausted (atom false)]
    (letfn [(dfs [i assign score]
              (when (> (swap! visited inc) budget) (throw (ex-info "budget" {:type ::budget})))
              (if (= i n)
                (let [w (count (violations-of nodes schema assign))
                      {bw :weight bs :score} @best]
                  (when (or (< w bw) (and (= w bw) (> score bs)))
                    (reset! best {:assign assign :weight w :score score})))
                (let [q (nth order i)]
                  (doseq [[l u] (utils q)]
                    (dfs (inc i) (assoc assign q l) (+ score u))))))]
      (try (dfs 0 {} 0.0)
           (catch Exception e
             (if (= ::budget (:type (ex-data e))) (reset! exhausted true) (throw e))))
      (let [assign (or (:assign @best) (into {} (map (fn [q] [q (ffirst (utils q))]) order)))]
        (result "min_violations" assign (score-of utils assign) (violations-of nodes schema assign)
                (not @exhausted))))))

(defn- primary
  "The chosen decoder's answer when it is feasible, else nil (__init__.py's
  _primary): independent when no constraint couples two questions, else
  exact with beam past the budget."
  [schema nodes utils {:keys [budget beam-width] :or {budget 200000 beam-width 16}}]
  (let [cross? (some #(> (count (references %)) 1) nodes)
        sol (if cross?
              (try (exact schema nodes utils budget)
                   (catch Exception e
                     (if (= ::budget (:type (ex-data e)))
                       (beam schema nodes utils beam-width)
                       (throw e))))
              (independent schema nodes utils))]
    (when (and sol (:feasible sol)) sol)))

(defn decode
  "Decide every question: schema (see `schema`), probs {qid [p ...]} in
  label order, nodes from `parse`. Answers {:assignment {qid label-index}
  :score :violations [constraint-index ...] :feasible :exact :decoder}.
  opts: :budget (200000 search nodes), :beam-width (16), :on-infeasible
  :min-violations (default) or :raise, which throws ex-info {:type
  :infeasible :violations [...]} with the minimal violation set."
  [schema probs nodes opts]
  (let [utils (utilities schema probs)]
    (or (primary schema nodes utils opts)
        (let [sol (min-violations schema nodes utils (:budget opts 200000))]
          (if (= :raise (:on-infeasible opts))
            (throw (ex-info "no assignment satisfies the constraints"
                            {:type :infeasible :violations (:violations sol) :assignment (:assignment sol)}))
            sol)))))
