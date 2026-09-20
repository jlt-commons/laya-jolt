;; In-distribution accuracy and latency on the trio localjev's bake-off uses:
;; AG News (choice, 4 topics), BoolQ (noul over a passage), SST-5 (score,
;; 5 levels), 40 balanced cases each, built by bench/triad120.py from the
;; public datasets (the texts are not checked in).
;;
;;   python3 bench/triad120.py                       # -> bench/data/triad120.jsonl
;;   jolt -M bench/triad.clj english
;;   jolt -M bench/triad.clj minicpm5 false           # a thinker, thinking off
;;   jolt -M bench/triad.clj minicpm5 true
;;   jolt -M bench/triad.clj english --backend mlx --dtype f16
(require '[lev.agent :as ag] '[lev.router :as router] '[lev.config :as cfg] '[lev.json :as json] '[clojure.string :as str])
(def cases (mapv json/read-str (remove str/blank? (str/split-lines (slurp "bench/data/triad120.jsonl")))))
;; positionals: model [thinking]; flags (--backend, --dtype, --data, --max-len ...)
;; anywhere. (A blind (drop 2 args) used to eat the first flag when there
;; was no thinking positional.)
(def opts (cfg/parse-args *command-line-args*))
(def model (or (first (:args opts)) "english"))
(def thinking (when (second (:args opts)) (= "true" (second (:args opts)))))
(def ctx (cfg/context opts))
(def rt (router/make-router {:data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
                             :thinkers (cfg/thinkers ctx)
                             :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                             :calibrations (cfg/calibrations ctx)}))
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
        mae (when (= "score" (get c "type")) (Math/abs (- (get a "score") (get c "expected"))))
        top (case (get c "type")
              "noul" (max (get a "noul") (- 1.0 (get a "noul")))
              (reduce max (vals (get a "probabilities"))))]
    {:task (get c "task") :type (get c "type") :ok ok :ms ms :mae mae :conf (get a "confidence") :top top}))

(defn ece [rs]
  (let [bins (group-by (fn [r] (min 9 (int (* 10 (:top r))))) rs)]
    (/ (reduce + (for [[_ xs] bins]
                   (* (count xs) (Math/abs (- (/ (count (filter :ok xs)) (double (count xs)))
                                              (/ (reduce + (map :top xs)) (double (count xs))))))))
       (double (max 1 (count rs))))))
(run1 (first cases)) (run1 (nth cases 50))
(def rs (mapv run1 cases))
(doseq [[task xs] (sort (group-by :task rs))]
  (println (format "  %-8s %3d/%3d  %5.1f%%  %s  %.0f ms/case" task (count (filter :ok xs)) (count xs) (* 100.0 (/ (count (filter :ok xs)) (count xs)))
                   (if (= task "sst5") (format "MAE %.2f" (/ (reduce + (map :mae xs)) (count xs))) "        ")
                   (/ (reduce + (map :ms xs)) (count xs)))))
(println (format "  %-8s %3d/%3d  %5.1f%%           %.0f ms/case   ECE %.3f" "all" (count (filter :ok rs)) (count rs) (* 100.0 (/ (count (filter :ok rs)) (count rs))) (/ (reduce + (map :ms rs)) (count rs)) (ece rs)))
(println "  gate by threshold: kept / accuracy kept / (accuracy the escalated part had), per task")
(doseq [thr [0.5 0.7 0.8 0.9]]
  (let [acc (fn [xs] (* 100.0 (/ (count (filter :ok xs)) (max 1 (count xs)))))]
    (println (format "    >= %.1f: %s" thr
                     (str/join "   " (for [[task xs] (sort (group-by :task rs))]
                                       (let [k (filter #(>= (:conf %) thr) xs) e (remove #(>= (:conf %) thr) xs)]
                                         (format "%s %2d/%d @ %5.1f%% (%5.1f%%)" task (count k) (count xs) (acc k) (acc e)))))))))
