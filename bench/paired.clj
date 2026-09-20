(ns bench.paired
  "Interleaved paired comparison of candidates: the honest way to measure
  an optimization. laya-mlx's experiments/engineering/paired.py and
  analyze.py, whose sequential pilot said 1.24x for a change the paired
  run put at 1.03x; the same drift shows in bench/workflow.clj, where the
  same code moves 288 -> 304 ms between two runs.

  Every round takes one input (the inputs rotate, so no candidate is
  timed on the same call twice running) and runs every candidate on it,
  in an order that rotates by one each round, so a drift in the machine's
  speed over the run lands on every candidate alike. Per round the ratio
  baseline-ms / candidate-ms is recorded; the report is the median of
  those ratios (the paired speedup) with a percentile-bootstrap 95%
  interval over the rounds (2,000 resamples, fixed seed) next to the
  p50s themselves. An interval that includes 1 is no win.

    jolt -M bench/paired.clj --candidates cpu,mlx,mlx16 [--case short4|long4|long1|all]
                             [--rounds 30] [--warmup 3] [--data DIR] [--model NAME]

  Candidates, the first the baseline: cpu (the C kernels), mlx (lev.mlx
  at f32, once jolt mlx has built it), mlx16 (lev.mlx at f16). `cpu,cpu`
  measures the noise floor: a ratio of 1 inside its interval, and how
  wide that interval is on this machine."
  (:require [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.mlx :as mlx]
            [lev.router :as router]))

;; --- statistics ---------------------------------------------------------------

(defn percentile
  "The value at fraction p of the sorted xs (nearest rank, p in [0, 1])."
  [xs p]
  (let [s (vec (sort xs))]
    (nth s (min (dec (count s)) (max 0 (int (Math/floor (* p (count s)))))))))

(defn median [xs]
  (let [s (vec (sort xs)) n (count s)]
    (if (even? n)
      (/ (+ (double (nth s (dec (quot n 2)))) (double (nth s (quot n 2)))) 2.0)
      (double (nth s (quot n 2))))))

(defn paired-ratios
  "baseline-ms / candidate-ms, round by round."
  [baseline candidate]
  (mapv (fn [b c] (/ (double b) (double c))) baseline candidate))

(defn- lcg
  "Park-Miller minimal standard: the next state after x (1 <= x < 2^31-1).
  Enough randomness for a bootstrap, no dependency, the same on any host."
  [x]
  (mod (* x 16807) 2147483647))

(defn paired-stats
  "The median paired ratio and its percentile-bootstrap 95% interval:
  `resamples` draws of (count ratios) rounds with replacement, the median
  of each, the 2.5th and 97.5th percentiles of those medians."
  [ratios {:keys [resamples seed] :or {resamples 2000 seed 20260919}}]
  (let [n (count ratios)
        medians (loop [i 0, x (inc (mod seed 2147483645)), acc (transient [])]
                  (if (= i resamples)
                    (persistent! acc)
                    (let [[x sample] (loop [j 0, x x, s (transient [])]
                                       (if (= j n)
                                         [x (persistent! s)]
                                         (let [x (lcg x)]
                                           (recur (inc j) x (conj! s (nth ratios (mod x n)))))))]
                      (recur (inc i) x (conj! acc (median sample))))))]
    {:median (median ratios)
     :interval [(percentile medians 0.025) (percentile medians 0.975)]
     :min (reduce min ratios)
     :max (reduce max ratios)
     :rounds n}))

;; --- rounds --------------------------------------------------------------------

(defn run-rounds
  "candidates: [[name (fn [input])] ...]; inputs: a vector the rounds cycle
  through. Warmup: every candidate on the first input, `warmup` times,
  untimed. Then `rounds` rounds: input (round mod inputs), the candidates
  in an order rotated by the round number, each timed alone. Answers
  {:samples {name [ms per round]}}."
  [candidates inputs {:keys [rounds warmup] :or {rounds 30 warmup 3}}]
  (when-not (apply distinct? (map first candidates))
    (throw (ex-info "candidate names must be distinct (they key the samples)"
                    {:type :invalid-candidates :names (mapv first candidates)})))
  (let [n (count candidates)]
    (dotimes [_ warmup]
      (doseq [[_ f] candidates] (f (first inputs))))
    (let [samples (reduce (fn [samples round]
                            (let [input (nth inputs (mod round (count inputs)))]
                              (reduce (fn [samples offset]
                                        (let [[name f] (nth candidates (mod (+ round offset) n))
                                              t0 (System/nanoTime)]
                                          (f input)
                                          (update samples name (fnil conj []) (/ (- (System/nanoTime) t0) 1e6))))
                                      samples
                                      (range n))))
                          {}
                          (range rounds))]
      {:samples samples})))

(defn compare-candidates
  "run-rounds, then every candidate after the first against it: {:samples
  {name [ms]} :p50 {name ms} :p95 {name ms} :ratios {name paired-stats}},
  ratios in candidate order."
  [candidates inputs {:keys [resamples seed] :as opts}]
  (let [{:keys [samples]} (run-rounds candidates inputs opts)
        [base & others] (map first candidates)
        stats (select-keys opts [:resamples :seed])]
    {:samples samples
     :p50 (into {} (map (fn [[name ms]] [name (median ms)])) samples)
     :p95 (into {} (map (fn [[name ms]] [name (percentile ms 0.95)])) samples)
     :ratios (apply array-map
                    (mapcat (fn [name] [name (paired-stats (paired-ratios (samples base) (samples name)) stats)])
                            others))}))

(defn report [label candidates {:keys [p50 p95 ratios]}]
  (println (format "  %s" label))
  (doseq [[name _] candidates]
    (println (format "    %-10s p50 %9.1f ms   p95 %9.1f ms" name (p50 name) (p95 name))))
  (doseq [[name {:keys [median interval min max rounds]}] ratios]
    (println (format "    %-10s paired speedup %.3fx  [%.3f, %.3f] 95%%   per-round %.3f..%.3f  (%d rounds)%s"
                     (str (ffirst candidates) "/" name) median (first interval) (second interval) min max rounds
                     (if (<= (first interval) 1.0 (second interval)) "  -- includes 1: no win" "")))))

;; --- the script ------------------------------------------------------------------

(load-file "bench/workload.clj")   ; jolt's load-file leaves *ns* in the file's
(in-ns 'bench.paired)
(alias 'wl 'bench.workload)

(defn distinct-names
  "names with every repeat suffixed #n (cpu, cpu -> cpu, cpu#2): the same
  agent measured against itself is the noise floor, and needs two series."
  [names]
  (first (reduce (fn [[out seen] name]
                   (let [k (inc (get seen name 0))]
                     [(conj out (if (= 1 k) name (str name "#" k))) (assoc seen name k)]))
                 [[] {}]
                 names)))

(defn- agent-for
  "The agent behind a candidate name, on the router's checkpoint."
  [rt model name]
  (case name
    "cpu" (router/load-model rt model)   ; the router keeps one agent per checkpoint: cpu,cpu is the same agent twice
    "mlx" (mlx/load-agent (get (:models rt) (router/normalise-name model)) {:name model :dtype :f32})
    "mlx16" (mlx/load-agent (get (:models rt) (router/normalise-name model)) {:name model :dtype :f16})
    (throw (ex-info (str "unknown candidate " (pr-str name) "; cpu, mlx or mlx16") {:type :unknown-candidate :name name}))))

(defn- usage []
  (binding [*out* *err*]
    (println "usage: jolt -M bench/paired.clj --candidates cpu,mlx,mlx16 [--case short4|long4|long1|all] [--rounds 30] [--warmup 3] [--data DIR] [--model NAME]")))

(defn -main [& args]
  (if (empty? args)
    (usage)
    (let [opts (cfg/parse-args args)
          ctx (cfg/context opts)
          names (str/split (or (get opts "--candidates") (do (usage) (System/exit 2))) #",")
          model (get opts "--model" "english")
          data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
          rounds (Long/parseLong (get opts "--rounds" "30"))
          warmup (Long/parseLong (get opts "--warmup" "3"))
          case-name (get opts "--case" "all")
          rt (router/make-router {:data data :max-loaded 3
                                  :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                  :calibrations (cfg/calibrations ctx)})
          agents (mapv (fn [n shown] [shown (agent-for rt model n)]) names (distinct-names names))
          cases (if (= "all" case-name) wl/cases (select-keys wl/cases [case-name]))]
      (when (< (count names) 2) (usage) (System/exit 2))
      (println (format "%s: %s, %d rounds after %d warmup, the input rotating over 8 state variants"
                       model (str/join " vs " names) rounds warmup))
      (doseq [[cname [state questions]] cases]
        (let [inputs (wl/variants state 8)
              candidates (mapv (fn [[n agent]] [n (fn [st] (ag/system-one agent st questions))]) agents)]
          (report (format "%s (%d questions)" cname (count questions)) candidates
                  (compare-candidates candidates inputs {:rounds rounds :warmup warmup})))))))

(apply -main *command-line-args*)
