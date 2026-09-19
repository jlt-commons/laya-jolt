(ns laya.sequence-test
  "json.dumps / build_sequence parity for laya.sequence.

  The state is serialized with json.dumps(state, ensure_ascii=False) BEFORE
  tokenization, so every byte of the JSON text is part of the parity surface:
  Python float repr (shortest round-trip, fixed for 1e-4 <= |x| < 1e16, else
  d.ddde+XX), control-char escapes (\\b \\f \\n \\r \\t, \\u00XX otherwise),
  and ensure_ascii=True for _to_internal's json.dumps(instructions).
  Expected strings below are the output of the CPython json module."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [laya.agent :as ag]
            [laya.sequence :as seq]
            [laya.test-util :as tu :refer [golden-dir data-dir]]
            [laya.tokenizer :as tk]))

(def tok (delay (tk/load (str data-dir "/tokenizer.edn"))))

(deftest py-float-repr
  (testing "fixed notation for -4 < decpt <= 16, exponent otherwise"
    (doseq [[x s] [[0.0001 "0.0001"] [0.00001 "1e-05"] [1e16 "1e+16"]
                   [1e15 "1000000000000000.0"] [123456789.0 "123456789.0"]
                   [1.0 "1.0"] [100.0 "100.0"] [0.5 "0.5"] [2.5e-7 "2.5e-07"]
                   [1e21 "1e+21"] [1e22 "1e+22"] [1.5e-10 "1.5e-10"]
                   [12345678901234567890.0 "1.2345678901234567e+19"]
                   [0.001 "0.001"] [0.000123 "0.000123"]
                   [9999999999999998.0 "9999999999999998.0"] [1e-7 "1e-07"]
                   [123.456 "123.456"] [-0.0001 "-0.0001"] [-1e-5 "-1e-05"]
                   [1.6369030475616455 "1.6369030475616455"] [160000.0 "160000.0"]
                   [(+ 0.1 0.2) "0.30000000000000004"] [5e-324 "5e-324"]
                   [1.7976931348623157e308 "1.7976931348623157e+308"]
                   [0.0 "0.0"] [-0.0 "-0.0"] [2.0 "2.0"] [1e7 "10000000.0"]
                   [9999999.0 "9999999.0"] [12345678.9 "12345678.9"]
                   [0.007 "0.007"] [0.9734 "0.9734"] [4.35 "4.35"]
                   [1e100 "1e+100"] [1.5e300 "1.5e+300"]
                   [123456789012345.6 "123456789012345.6"]]]
      (is (= s (seq/py-float-str x)) (str x))))
  (testing "non-finite, as json.dumps writes them"
    (is (= "NaN" (seq/py-float-str ##NaN)))
    (is (= "Infinity" (seq/py-float-str ##Inf)))
    (is (= "-Infinity" (seq/py-float-str ##-Inf)))))

(deftest json-str-matches-json-dumps
  (testing "ensure_ascii=False (serialize_state)"
    (is (= "{\"a\": [1, 2.0, true, null, \"x\\by\\fzé😀 \u007f\\u0001\"]}"
           (seq/json-str {"a" [1 2.0 true nil "x\by\fzé😀 \u007f\u0001"]})))
    (is (= "{\"k\": {\"n\": [], \"m\": {}}, \"s\": \"q\\\"\\\\/\"}"
           (seq/json-str (array-map "k" (array-map "n" [] "m" {}) "s" "q\"\\/")))))
  (testing "ensure_ascii=True (_to_internal's json.dumps(instructions))"
    (is (= "{\"a\": [1, 2.0, true, null, \"x\\by\\fz\\u00e9\\ud83d\\ude00\\u2028\\u007f\\u0001\"]}"
           (seq/json-str {"a" [1 2.0 true nil "x\by\fzé😀 \u007f\u0001"]}
                         {:ensure-ascii true}))))
  (testing "floats inside documents use the Python repr"
    (is (= "{\"amount\": 0.0001, \"big\": 1e+16, \"n\": 3}"
           (seq/json-str (array-map "amount" 0.0001 "big" 1e16 "n" 3)))))
  (testing "keywords serialize by name, like the string keys Python sees"
    (is (= "{\"type\": \"choice\"}" (seq/json-str {:type :choice})))))

(deftest temp-bucket-cardinalities
  (is (= "choice:2" (seq/temp-bucket 0 1)))
  (is (= "choice:2" (seq/temp-bucket 0 2)))
  (is (= "score:3-5" (seq/temp-bucket 1 3)))
  (is (= "score:3-5" (seq/temp-bucket 1 5)))
  (is (= "noul:6-10" (seq/temp-bucket 2 6)))
  (is (= "choice:6-10" (seq/temp-bucket 0 10)))
  (is (= "choice:11+" (seq/temp-bucket 0 11)))
  (is (= "choice:11+" (seq/temp-bucket 0 200))))

(deftest ordered-map-survives-growth
  (let [ks (map str (range 30))]
    (is (= ks (keys (seq/ordered-map (map vector ks (range 30))))))
    (is (= "{\"z\": 1, \"a\": 2}" (seq/json-str (seq/ordered-map [["z" 1] ["a" 2]]))))))

(deftest render-options-falsy-values
  (testing "python `if not v`: None, False, 0, 0.0, '', empty collections"
    (is (= ["a" "b" "c" "d" "e" "f" "g: 1"]
           (seq/render-options {:t "choice"
                                :crit (array-map "a" nil "b" false "c" 0 "d" 0.0
                                                 "e" "" "f" [] "g" 1)}))))
  (testing "score levels and noul defaults"
    (is (= ["level 0: low" "level 1: high"]
           (seq/render-options {:t "score" :crit ["low" "high"]})))
    (is (= ["false: no, the statement does not hold" "true: yes, the statement holds"]
           (seq/render-options {:t "noul" :crit nil})))
    (is (= ["false: legit" "true: scam"]
           (seq/render-options {:t "noul" :crit {"true" "scam" "false" "legit"}})))))

(deftest build-sequence-matches-oracle
  (testing "README quickstart ids + markers, straight from golden/readme.edn"
    (let [cases (tu/read-golden "cases")
          readme (tu/read-golden "readme")
          state (:readme-state cases)]
      (doseq [[qid qdef] (:readme-questions cases)]
        (let [q (ag/to-internal qdef)
              [ids markers] (seq/build-sequence @tok state q 512 192)
              gold (get (:input-ids readme) qid)]
          (is (= (mapv long (:ids gold)) ids) qid)
          (is (= (mapv long (:markers gold)) markers) qid)
          (is (= (:qtype gold) (seq/qtypes (:t q))) qid)))))
  (testing "the branches the quickstart never takes (golden/sequences.edn)"
    (let [cases (tu/read-golden "sequences")]
      (is (= 12 (count cases)))
      (doseq [[name {:keys [state question ids markers]}] cases]
        (let [q (ag/to-internal question)
              [got-ids got-markers] (seq/build-sequence @tok state q 512 192)]
          (is (= (mapv long ids) got-ids) name)
          (is (= (mapv long markers) got-markers) name)))
      (testing "what those cases pin down"
        (let [ids (fn [k] (mapv long (get-in cases [k :ids])))]
          (is (= 512 (count (ids "long-state-truncates"))) "hard max_len cut")
          (is (= 50282 (peek (ids "long-state-truncates"))) "...but the final [SEP] survives")
          (is (= 12 (count (get-in cases ["many-long-options-shrink" :markers]))) "12 options shrunk evenly all fit")
          (is (= 2 (count (filter #{50284} (ids "mask-injection"))))
              "[MASK] in the state and instructions is neutralized: only the two option markers remain")
          (is (= 3 (count (filter #{50281} (ids "mask-injection"))))
              "...while [CLS] in the state stays a special token (leading CLS + two in the state)")
          (is (= 11 (count (get-in cases ["score-many-levels" :markers])))))))))
