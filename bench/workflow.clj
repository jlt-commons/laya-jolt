(ns bench.workflow
  "Latency of a multi-question call, the shape the prefix cache and the
  once-per-call state tokenization are for: the README's four email
  questions (choice, score, noul, noul) over a short state (the README's
  email, 53 tokens) and over a long one (a 1.6k-token email, cut to
  max_len for every question), from bench/workload.clj. Every call repeats the same questions on
  the same state, as a workflow does; the report is p50 / p95 over the
  iterations after a warmup, and the tokenization alone for the long state.

    jolt -M bench/workflow.clj [--data DIR] [--model NAME] [--iterations N] [--warmup N]

  --model may name a thinker (config.edn :thinkers, LEV_THINKER): then
  the calls go through lev.think, with thinking off unless the thinker's
  config turns it on, and the state tokenization lines are left out;
  two more rows time 8 short states (the email with \"Record i: \" in
  front) one call each against one lev.agent/system-one-batch.

  No answers change under either optimization (golden/ pins them); this
  measures the time they take."
  (:require [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.router :as router]
            [lev.sequence :as seq]))

(load-file "bench/workload.clj")   ; jolt's load-file leaves *ns* in the file's
(in-ns 'bench.workflow)
(alias 'wl 'bench.workload)

(defn- percentile [xs p]
  (let [s (vec (sort xs))]
    (nth s (min (dec (count s)) (int (Math/floor (* p (count s))))))))

(defn- timed [f iterations warmup]
  (dotimes [_ warmup] (f))
  (let [ms (vec (repeatedly iterations #(let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6))))]
    {:p50 (percentile ms 0.5) :p95 (percentile ms 0.95) :mean (/ (reduce + ms) (count ms))}))

(defn- row [label {:keys [p50 p95 mean]}]
  (println (format "  %-44s p50 %8.1f ms   p95 %8.1f ms   mean %8.1f ms" label p50 p95 mean)))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        model (get opts "--model" "english")
        data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
        iterations (Long/parseLong (get opts "--iterations" "30"))
        warmup (Long/parseLong (get opts "--warmup" "3"))
        rt (router/make-router {:data data
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)
                                :thinkers (cfg/thinkers ctx)})
        agent (router/load-model rt model)
        tok (:tok agent)]
    (println (format "%s, %d questions a call, %d iterations after %d warmup" model (count wl/questions) iterations warmup))
    (when tok
      (println (format "  short state: %d tokens; long state: %d tokens (max_len %d)"
                       (count (seq/encode-state tok wl/short-state)) (count (seq/encode-state tok wl/long-state))
                       (:max-len (:cfg agent))))
      (row "tokenize the long state once" (timed #(seq/encode-state tok wl/long-state) iterations warmup)))
    (row "4 questions, short state" (timed #(ag/system-one agent wl/short-state wl/questions) iterations warmup))
    (row "4 questions, long state" (timed #(ag/system-one agent wl/long-state wl/questions) iterations warmup))
    (row "1 question (department), long state" (timed #(ag/system-one agent wl/long-state (select-keys wl/questions ["department"])) iterations warmup))
    (when (= :thinker (:kind agent))
      (let [states (wl/variants wl/short-state 8)]
        (row "8 short states, one call each" (timed #(mapv (fn [s] (ag/system-one agent s wl/questions)) states) iterations warmup))
        (row "8 short states, one batch" (timed #(ag/system-one-batch agent states wl/questions) iterations warmup))))))

(apply -main *command-line-args*)
