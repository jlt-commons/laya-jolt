(ns bench.workflow
  "Latency of a multi-question call, the shape the prefix cache and the
  once-per-call state tokenization are for: the README's four email
  questions (choice, score, noul, noul) over a short state (the README's
  email, 367 input tokens) and over a long one (a 1.4k-token email, cut to
  max_len for every question). Every call repeats the same questions on
  the same state, as a workflow does; the report is p50 / p95 over the
  iterations after a warmup, and the tokenization alone for the long state.

    jolt -M bench/workflow.clj [--data DIR] [--model NAME] [--iterations N] [--warmup N]

  No answers change under either optimization (golden/ pins them); this
  measures the time they take."
  (:require [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.router :as router]
            [lev.sequence :as seq]))

(def questions
  (array-map
   "department" {"type" "choice"
                 "instructions" "Which department should handle this email?"
                 "criteria" (array-map "billing" "invoices, payments, refunds"
                                       "technical" "bugs, outages, integrations"
                                       "sales" "pricing, contracts, demos"
                                       "other" "everything else")}
   "urgency" {"type" "score"
              "instructions" "How urgent is this request?"
              "criteria" ["not urgent" "soon" "critical deadline or blocking issue"]}
   "churn_risk" {"type" "noul"
                 "instructions" "Does the user threaten to cancel or switch to a competitor?"}
   "is_phishing" {"type" "noul"
                  "instructions" "Is this email a phishing or scam attempt?"}))

(def short-state
  (array-map
   "from" "customer@acme.com"
   "subject" "Duplicate billing on March invoice #4411"
   "body" "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan."))

(def long-state
  (assoc short-state "body"
         (apply str (repeat 40 "We were billed twice for the same invoice INV-2291 last Tuesday and the support line was closed; please refund the duplicate charge to the card ending 4412 before Friday, or we cancel. "))))

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
                                :calibrations (cfg/calibrations ctx)})
        agent (router/load-model rt model)
        tok (:tok agent)]
    (println (format "%s, %d questions a call, %d iterations after %d warmup" model (count questions) iterations warmup))
    (println (format "  short state: %d tokens; long state: %d tokens (max_len %d)"
                     (count (seq/encode-state tok short-state)) (count (seq/encode-state tok long-state))
                     (:max-len (:cfg agent))))
    (row "tokenize the long state once" (timed #(seq/encode-state tok long-state) iterations warmup))
    (row "4 questions, short state" (timed #(ag/system-one agent short-state questions) iterations warmup))
    (row "4 questions, long state" (timed #(ag/system-one agent long-state questions) iterations warmup))
    (row "1 question (department), long state" (timed #(ag/system-one agent long-state (select-keys questions ["department"])) iterations warmup))))

(apply -main *command-line-args*)
