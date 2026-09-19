(ns laya.test-util
  "Shared helpers for the parity suites."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.test :refer [is]]
            [laya.agent :as ag]
            [laya.tensors]
            [laya.sequence :as seq]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LAYA_DATA") "data"))

(def agent
  "The english checkpoint, loaded once for the whole suite: every load is
  ~1.7 GB of f32, and the CI mac runner has 7 GB."
  (delay (ag/load-agent data-dir)))

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

(def accelerate?
  "Apple's Accelerate is the BLAS the goldens are byte-identical under. On
  x86 OpenBLAS picks kernels per CPU model, and the ubuntu CI runners are
  not all the same machine: the same weights gave a 4-decimal probability
  of 0.3142 on one and 0.3143 on another, a value sitting on the rounding
  boundary. So answers are exact on mac and within one unit in the last
  place elsewhere."
  (some? (re-find #"^Mac" (System/getProperty "os.name"))))

(defn answers-match
  "Assert a system-one JSON string against its golden: parsed and within
  1e-4 everywhere, byte-for-byte under Accelerate."
  [want got & [msg]]
  (is (approx= 1.0001e-4 (json/read-str want) (json/read-str got)) (or msg "answers within 1e-4"))
  (when accelerate?
    (is (= want got) (or msg "byte-identical under Accelerate"))))

(defn relative-max-abs
  "max |a-b| over the first n values, as a fraction of max |b|: the bound
  the residual stream needs (outlier dims reach ~3e4, where an f32 ulp is
  2e-3, and every sgemm summation order lands a few ulps apart)."
  [a b n]
  (let [pa (laya.tensors/ptr a) pb (laya.tensors/ptr b)]
    (loop [i 0 diff 0.0 scale 0.0]
      (if (= i n)
        [(/ diff (max scale 1e-30)) diff scale]
        (let [x (laya.tensors/get pa i) y (laya.tensors/get pb i)]
          (recur (inc i) (max diff (Math/abs (- x y))) (max scale (Math/abs y))))))))
