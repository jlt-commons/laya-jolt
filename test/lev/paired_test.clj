(ns lev.paired-test
  "bench/paired.clj's statistics and round discipline, on synthetic timings:
  the harness decides what lands (CLAUDE.md: anything that changes numbers
  is measured), so its arithmetic is tested. The script is loaded with no
  command-line arguments, so its entry point prints usage and returns."
  (:require [clojure.test :refer [deftest is testing]]))

(binding [*command-line-args* nil
          *err* (java.io.StringWriter.)]
  (load-file "bench/paired.clj"))
(in-ns 'lev.paired-test)   ; jolt's load-file leaves *ns* in the file's

(alias 'p 'bench.paired)

(deftest medians-and-percentiles
  (is (= 3.0 (p/median [5 1 3 2 4])))
  (is (= 2.5 (p/median [4 1 3 2])))
  (is (= 1 (p/percentile [1 2 3 4 5] 0.0)))
  (is (= 5 (p/percentile [1 2 3 4 5] 1.0)))
  (is (= 3 (p/percentile [5 4 3 2 1] 0.5))))

(deftest paired-ratios-are-per-round
  (testing "round i compares the baseline's i-th sample with the candidate's i-th"
    (is (= [2.0 2.0 0.5] (p/paired-ratios [10.0 20.0 5.0] [5.0 10.0 10.0])))))

(deftest bootstrap-interval-brackets-the-median
  (testing "a candidate exactly twice as fast in every round: the median is 2 and the
            interval is [2 2] whatever the resample"
    (let [ratios (vec (repeat 30 2.0))
          {:keys [median interval]} (p/paired-stats ratios {:resamples 500 :seed 1})]
      (is (= 2.0 median))
      (is (= [2.0 2.0] interval))))
  (testing "ratios scattered around 1.5 with noise: the interval holds the median,
            is narrower than the sample range, and is the same for the same seed"
    (let [ratios (mapv (fn [i] (+ 1.5 (* 0.2 (Math/sin (* 7.0 i))))) (range 40))
          a (p/paired-stats ratios {:resamples 2000 :seed 20260919})
          b (p/paired-stats ratios {:resamples 2000 :seed 20260919})
          [lo hi] (:interval a)]
      (is (= a b) "deterministic")
      (is (<= lo (:median a) hi))
      (is (< (reduce min ratios) lo))
      (is (> (reduce max ratios) hi))
      (is (< (- hi lo) 0.2))))
  (testing "a candidate no faster: the interval includes 1"
    (let [ratios (mapv (fn [i] (+ 1.0 (* 0.05 (Math/cos (* 3.0 i))))) (range 32))
          [lo hi] (:interval (p/paired-stats ratios {:resamples 1000 :seed 3}))]
      (is (<= lo 1.0 hi)))))

(deftest rounds-rotate-candidates-and-inputs
  (let [calls (atom [])
        cand (fn [name] [name (fn [input] (swap! calls conj [name input]) nil)])
        result (p/run-rounds [(cand "a") (cand "b") (cand "c")] [:x :y] {:rounds 4 :warmup 1})]
    (testing "warmup: every candidate once on the first input, untimed"
      (is (= [["a" :x] ["b" :x] ["c" :x]] (take 3 @calls))))
    (testing "each round: one input, every candidate on it, the order rotating by one"
      (is (= [["a" :x] ["b" :x] ["c" :x]
              ["b" :y] ["c" :y] ["a" :y]
              ["c" :x] ["a" :x] ["b" :x]
              ["a" :y] ["b" :y] ["c" :y]]
             (drop 3 @calls))))
    (testing "one timing per candidate per round"
      (is (= #{"a" "b" "c"} (set (keys (:samples result)))))
      (is (every? #(= 4 (count %)) (vals (:samples result))))
      (is (every? #(every? (fn [ms] (and (number? ms) (>= ms 0))) %) (vals (:samples result)))))))

(deftest candidate-names-must-be-distinct
  (testing "two candidates with one name would share a sample series and compare as 1.000x"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"distinct"
                          (p/run-rounds [["cpu" identity] ["cpu" identity]] [:x] {:rounds 1 :warmup 0}))))
  (testing "the script's names are made distinct: a repeated name gets a #n suffix"
    (is (= ["cpu" "cpu#2" "mlx" "cpu#3"] (p/distinct-names ["cpu" "cpu" "mlx" "cpu"])))))

(deftest compare-reports-every-candidate-against-the-first
  (let [slow (fn [_] (let [t0 (System/nanoTime)] (while (< (- (System/nanoTime) t0) 2000000))))
        fast (fn [_] (let [t0 (System/nanoTime)] (while (< (- (System/nanoTime) t0) 500000))))
        report (p/compare-candidates [["slow" slow] ["fast" fast] ["same" slow]] [:only]
                                     {:rounds 12 :warmup 1 :resamples 300 :seed 5})]
    (is (= ["fast" "same"] (keys (:ratios report))))
    (let [{:keys [median interval]} (get-in report [:ratios "fast"])]
      (is (< 2.5 median 6.0) (str "a 2 ms vs 0.5 ms candidate: " median))
      (is (< 1.0 (first interval))))
    (let [[lo hi] (get-in report [:ratios "same" :interval])]
      (is (<= lo 1.05))
      (is (>= hi 0.95)))))
