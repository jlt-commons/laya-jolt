(ns laya.agent
  "RLAgent.system_one: state + typed questions -> calibrated typed answers.
  Mirrors rl_agent_api.py (temperature calibration, jev confidence)."
  (:require [clojure.edn :as edn]
            [laya.model :as m]
            [laya.sequence :as seq]
            [laya.tokenizer :as tk]))

(defn load-agent
  "Load config + tokenizer + all weights from data-dir."
  [data-dir]
  (let [manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        tok (tk/load (str data-dir "/tokenizer.edn"))]
    {:cfg cfg :tok tok :w (m/load-weights manifest data-dir)}))

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

(defn- round4 [x] (/ (double (Math/round (* 1e4 (double x)))) 1e4))

(defn- softmax [xs]
  (let [mx (reduce max xs)
        ex (mapv #(Math/exp (- (double %) mx)) xs)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn- confidence
  "1 - normalized entropy (confidence_from_probs)."
  [p k]
  (if (< k 2)
    1.0
    (let [ent (- (reduce + (map (fn [pi] (* pi (Math/log (max pi 1e-12)))) p)))]
      (- 1 (/ ent (Math/log k))))))

(defn- argmax [xs]
  (reduce (fn [bi i] (if (> (nth xs i) (nth xs bi)) i bi)) 0 (range (count xs))))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- answer-for [q p k actp]
  (let [ext (array-map "act_probability" actp)]
    (case (:t q)
      "choice"
      (let [ks (mapv key-str (keys (:crit q)))]
        (array-map "type" "choice"
                   "choice" (nth ks (argmax p))
                   "probabilities" (seq/ordered-map (map-indexed (fn [i c] [c (round4 (nth p i))]) ks))
                   "confidence" (round4 (confidence p k))
                   "rl_agent" ext))
      "score"
      (array-map "type" "score"
                 "score" (round4 (reduce + (map-indexed (fn [i pi] (* i pi)) p)))
                 "legend" (seq/ordered-map (map-indexed (fn [i c] [(str i) c]) (:crit q)))
                 "probabilities" (seq/ordered-map (map-indexed (fn [i pi] [(str i) (round4 pi)]) p))
                 "confidence" (round4 (confidence p k))
                 "rl_agent" ext)
      (array-map "type" "noul"
                 "noul" (round4 (nth p 1))
                 "rl_agent" ext))))

(defn system-one
  "state + {qid -> qdef} -> Jev answer map (ordered to match json.dumps)."
  [agent state questions]
  (let [{:keys [cfg tok w]} agent
        qids (vec (keys questions))
        prepared (mapv (fn [qid]
                         (let [q (to-internal (validate-question qid (get questions qid)))
                               [ids markers] (seq/build-sequence tok state q
                                                                 (:max-len cfg)
                                                                 (:head-max-len cfg))]
                           (when (not= (count markers) (count (seq/render-options q)))
                             (throw (ex-info (format "question %s: options do not fit in head_max_len=%d tokens"
                                                     (pr-str qid) (:head-max-len cfg))
                                             {:type :invalid-question :qid qid :field "criteria"
                                              :head-max-len (:head-max-len cfg)})))
                           {:qid qid :q q :ids ids :markers markers :qtype (seq/qtypes (:t q))}))
                       qids)
        n-tokens (reduce + (map #(count (:ids %)) prepared))
        answers (seq/ordered-map
                 (for [{:keys [qid q ids markers qtype]} prepared]
                   (let [k (count markers)
                         [logits act] (m/forward-row w cfg ids (vec (repeat (count ids) 1))
                                                     markers (vec (repeat k 1)) qtype)
                         temp (get (:temperature-by-options cfg)
                                   (seq/temp-bucket qtype k)
                                   (nth (:temperature cfg) qtype))
                         p (softmax (mapv #(/ (double %) temp) (take k logits)))
                         actp (first (softmax act))]
                     [qid (answer-for q p k actp)])))]
    (array-map "model" "rl-agent"
               "answers" answers
               "usage" (array-map "input_tokens" n-tokens "output_tokens" 0))))
