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

(defn- to-internal
  "Jev question def -> {:t :ins :crit} (rl_agent_api.RLAgent._to_internal)."
  [qdef]
  (let [t (:type qdef)
        crit (:criteria qdef)
        crit (if (and (= t "choice") (vector? crit))
               (into (array-map) (map (fn [c] [c nil])) crit)
               crit)]
    {:t t
     :ins (let [ins (:instructions qdef)]
            (if (string? ins) ins (seq/json-str ins)))
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
                   "probabilities" (into (array-map)
                                         (map-indexed (fn [i c] [c (round4 (nth p i))]) ks))
                   "confidence" (round4 (confidence p k))
                   "rl_agent" ext))
      "score"
      (array-map "type" "score"
                 "score" (round4 (reduce + (map-indexed (fn [i pi] (* i pi)) p)))
                 "legend" (into (array-map) (map-indexed (fn [i c] [(str i) c]) (:crit q)))
                 "probabilities" (into (array-map)
                                       (map-indexed (fn [i pi] [(str i) (round4 pi)]) p))
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
                         (let [q (to-internal (get questions qid))
                               [ids markers] (seq/build-sequence tok state q
                                                                 (:max-len cfg)
                                                                 (:head-max-len cfg))]
                           {:qid qid :q q :ids ids :markers markers :qtype (seq/qtypes (:t q))}))
                       qids)
        n-tokens (reduce + (map #(count (:ids %)) prepared))
        answers (reduce
                 (fn [acc {:keys [qid q ids markers qtype]}]
                   (let [k (count markers)
                         [logits act] (m/forward-row w cfg ids (vec (repeat (count ids) 1))
                                                     markers (vec (repeat k 1)) qtype)
                         temp (get (:temperature-by-options cfg)
                                   (seq/temp-bucket qtype k)
                                   (nth (:temperature cfg) qtype))
                         p (softmax (mapv #(/ (double %) temp) (take k logits)))
                         actp (first (softmax act))]
                     (assoc acc qid (answer-for q p k actp))))
                 (array-map)
                 prepared)]
    (array-map "model" "rl-agent"
               "answers" answers
               "usage" (array-map "input_tokens" n-tokens "output_tokens" 0))))
