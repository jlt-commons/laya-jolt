(ns laya.json-test
  "laya.json: the order-preserving JSON reader behind the HTTP API. Key
  order on the wire is model input (options, questions, state fields), and
  clojure.data.json builds hash-maps, which drop it past 8 keys."
  (:require [clojure.test :refer [deftest is testing]]
            [laya.json :as json]
            [laya.sequence :as seq]))

(deftest scalars-and-nesting
  (is (= {"a" 1 "b" [true false nil] "c" "s"} (json/read-str "{\"a\": 1, \"b\": [true, false, null], \"c\": \"s\"}")))
  (is (= [] (json/read-str " [ ] ")))
  (is (= {} (json/read-str "{}")))
  (is (= [[[1]]] (json/read-str "[[[1]]]"))))

(deftest objects-keep-key-order-at-any-size
  (let [ks (map #(str "k" %) (range 40))
        doc (str "{" (clojure.string/join ", " (map #(str "\"" % "\": " %2) ks (range))) "}")]
    (is (= ks (keys (json/read-str doc))))
    (is (= ["z" "a" "m"] (keys (json/read-str "{\"z\":1,\"a\":2,\"m\":3}"))))
    (testing "round trip through json-str keeps the order"
      (is (= doc (seq/json-str (json/read-str doc)))))))

(deftest numbers-like-python-json
  (is (= 8192 (json/read-str "8192")))
  (is (integer? (json/read-str "8192")))
  (is (= -3 (json/read-str "-3")))
  (is (= 160000.0 (json/read-str "160000.0")))
  (is (= 1.0E-5 (json/read-str "1e-05")))
  (is (= 1.6369030475616455 (json/read-str "1.6369030475616455")))
  (is (= 1.0E21 (json/read-str "1E21")))
  (is (float? (json/read-str "2e3")))
  (is (= 12345678901234567890 (json/read-str "12345678901234567890")) "big ints"))

(deftest strings-and-escapes
  (is (= "a\"b\\c/d\b\f\n\r\t" (json/read-str "\"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\"")))
  (is (= "é😀" (json/read-str "\"\\u00e9\\ud83d\\ude00\"")) "BMP + surrogate pair")
  (is (= "Ġ😀" (json/read-str "\"Ġ😀\"")) "raw non-ascii passes through")
  (is (= "" (json/read-str "\"\""))))

(deftest errors
  (doseq [bad ["{\"a\": }" "[1,]" "tru" "{} x" "" "{\"a\" 1}" "[1 2]" "\"unterminated" "{\"a\": 1,}" "nul" "01x"]]
    (is (thrown-with-msg? Exception #"^json:" (json/read-str bad)) (pr-str bad))))
