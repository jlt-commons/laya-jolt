(ns lev.test-util
  "Shared helpers for the parity suites."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.test :refer [is]]
            [lev.agent :as ag]
            [lev.tensors]
            [lev.sequence :as seq]))

(def golden-dir
  (or (System/getenv "LEV_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LEV_DATA") "data"))

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

(defn answers-match
  "Assert a system-one JSON string against its golden: parsed and within
  one unit in the fourth decimal everywhere.

  Not byte-for-byte: answers are rounded to 4 decimals and sgemm summation
  order differs by BLAS, by CPU model under OpenBLAS, and by how attention
  is blocked, so a value on a rounding boundary flips its last digit. The
  quickstart's urgency p[1] is one (0.314250x): torch printed 0.3142, the
  per-head sgemm attention under Accelerate lands at 0.3143, and the ubuntu
  runners have given both."
  [want got & [msg]]
  ;; the goldens carry the Python package's own model name; lev reports the
  ;; checkpoint's name (english, ...) or "encoder" for an agent loaded alone
  (is (approx= 1.0001e-4 (dissoc (json/read-str want) "model") (dissoc (json/read-str got) "model"))
      (or msg "answers within 1e-4")))

(defn relative-max-abs
  "max |a-b| over the first n values, as a fraction of max |b|: the bound
  the residual stream needs (outlier dims reach ~3e4, where an f32 ulp is
  2e-3, and every sgemm summation order lands a few ulps apart)."
  [a b n]
  (let [pa (lev.tensors/ptr a) pb (lev.tensors/ptr b)]
    (loop [i 0 diff 0.0 scale 0.0]
      (if (= i n)
        [(/ diff (max scale 1e-30)) diff scale]
        (let [x (lev.tensors/get pa i) y (lev.tensors/get pb i)]
          (recur (inc i) (max diff (Math/abs (- x y))) (max scale (Math/abs y))))))))
