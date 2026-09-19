(ns laya.test-util
  "Shared helpers for the parity suites."
  (:require [clojure.edn :as edn]
            [laya.sequence :as seq]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LAYA_DATA") "data"))

(def readers
  "The golden dumps write dicts as #laya/omap [[k v] ...] so that
  insertion order survives past 8 keys (an EDN map literal would not)."
  {'laya/omap seq/ordered-map})

(defn read-golden
  "Parse golden/<name>.edn."
  [name]
  (edn/read-string {:readers readers} (slurp (str golden-dir "/" name ".edn"))))

(defn approx=
  "Recursive equality with an absolute tolerance on numbers; everything
  else (strings, keys, structure) must be identical."
  [tol a b]
  (cond
    (and (number? a) (number? b)) (<= (Math/abs (- (double a) (double b))) tol)
    (and (map? a) (map? b)) (and (= (set (keys a)) (set (keys b)))
                                 (every? (fn [[k v]] (approx= tol v (get b k))) a))
    (and (sequential? a) (sequential? b)) (and (= (count a) (count b))
                                               (every? true? (map #(approx= tol %1 %2) a b)))
    :else (= a b)))
