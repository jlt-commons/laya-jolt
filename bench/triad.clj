;; In-distribution accuracy and latency on the trio localjev's bake-off uses:
;; AG News (choice, 4 topics), BoolQ (noul over a passage), SST-5 (score,
;; 5 levels), 40 balanced cases each, built by bench/triad120.py from the
;; public datasets (the texts are not checked in).
;;
;;   python3 bench/triad120.py                       # -> bench/data/triad120.jsonl
;;   jolt -M bench/triad.clj english
;;   jolt -M bench/triad.clj minicpm5 false           # a thinker, thinking off
;;   jolt -M bench/triad.clj minicpm5 true
(require '[lev.agent :as ag] '[lev.router :as router] '[lev.config :as cfg] '[lev.json :as json] '[clojure.string :as str])
(def cases (mapv json/read-str (remove str/blank? (str/split-lines (slurp "bench/data/triad120.jsonl")))))
(def model (or (first *command-line-args*) "english"))
(def thinking (when (second *command-line-args*) (= "true" (second *command-line-args*))))
(def rt (router/make-router {:data "data" :thinkers (cfg/thinkers (cfg/context {}))}))
(def agent (router/load-model rt model))
(defn q [c] (case (get c "type")
              "choice" (array-map "type" "choice" "instructions" (get c "instructions") "criteria" (get c "criteria"))
              "noul" (array-map "type" "noul" "instructions" (get c "instructions"))
              "score" (array-map "type" "score" "instructions" (get c "instructions") "criteria" (get c "criteria"))))
(defn run1 [c]
  (let [t0 (System/nanoTime)
        out (ag/system-one agent (get c "state") {"q" (q c)} (when (some? thinking) {:thinking thinking}))
        a (get-in out ["answers" "q"])
        ms (/ (- (System/nanoTime) t0) 1e6)
        pred (case (get c "type")
               "choice" (get a "choice")
               "noul" (>= (get a "noul") 0.5)
               "score" (let [p (get a "probabilities")] (apply max-key #(get p (str %)) (range (count (get c "criteria"))))))
        ok (= pred (get c "expected"))
        mae (when (= "score" (get c "type")) (Math/abs (- (get a "score") (get c "expected"))))]
    {:task (get c "task") :ok ok :ms ms :mae mae}))
(run1 (first cases)) (run1 (nth cases 50))
(def rs (mapv run1 cases))
(doseq [[task xs] (sort (group-by :task rs))]
  (println (format "  %-8s %3d/%3d  %5.1f%%  %s  %.0f ms/case" task (count (filter :ok xs)) (count xs) (* 100.0 (/ (count (filter :ok xs)) (count xs)))
                   (if (= task "sst5") (format "MAE %.2f" (/ (reduce + (map :mae xs)) (count xs))) "        ")
                   (/ (reduce + (map :ms xs)) (count xs)))))
(println (format "  %-8s %3d/%3d  %5.1f%%           %.0f ms/case" "all" (count (filter :ok rs)) (count rs) (* 100.0 (/ (count (filter :ok rs)) (count rs))) (/ (reduce + (map :ms rs)) (count rs))))
