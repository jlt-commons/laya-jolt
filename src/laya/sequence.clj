(ns laya.sequence
  "build_sequence + option rendering, mirroring rl_common.py.

  Sequence layout:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  The [MASK] positions (one per option, in label order) are returned as markers."
  (:require [clojure.string :as str]
            [laya.tokenizer :as tk]))

(def qtypes {"choice" 0 "score" 1 "noul" 2})
(def qtype-names {0 "choice" 1 "score" 2 "noul"})

(defn ordered-map
  "An insertion-ordered map from [k v] pairs, however many. Python dicts
  keep insertion order and both option order and answer order are part of
  the contract; `into`/`assoc` on an array-map silently become a hash-map
  past 8 entries, `(apply array-map ...)` does not."
  [kvs]
  (apply array-map (mapcat identity kvs)))

(defn py-float-str
  "Python float.__repr__ / json.dumps of a double: the shortest round-trip
  digits, fixed notation when 1e-4 <= |x| < 1e16, else d.ddde+XX with a
  signed two-digit-minimum exponent. json.dumps spells the non-finite values
  NaN / Infinity / -Infinity."
  [x]
  (let [x (double x)]
    (cond
      (Double/isNaN x) "NaN"
      (= x ##Inf) "Infinity"
      (= x ##-Inf) "-Infinity"
      (zero? x) (if (neg? (/ 1.0 x)) "-0.0" "0.0")
      :else
      ;; Chez prints the same shortest digits, Java-style: d.ddd or d.dddE[-]n.
      (let [s (str x)
            minus? (str/starts-with? s "-")
            s (if minus? (subs s 1) s)
            [mant e] (str/split s #"E")
            [ip fp] (str/split mant #"\.")
            fp (or fp "")
            e (if e (Long/parseLong e) 0)
            raw (str ip fp)
            lead (count (take-while #(= % \0) raw))
            digits (str/replace (subs raw lead) #"0+$" "")
            digits (if (= "" digits) "0" digits)
            ;; digits before the decimal point when value = 0.D x 10^decpt
            decpt (- (+ (count ip) e) lead)
            nd (count digits)
            body (if (or (<= decpt -4) (> decpt 16))
                   (let [ex (dec decpt)]
                     (str (subs digits 0 1)
                          (when (> nd 1) (str "." (subs digits 1)))
                          "e" (if (neg? ex) "-" "+")
                          (format "%02d" (Math/abs ex))))
                   (cond
                     (<= decpt 0) (str "0." (apply str (repeat (- decpt) "0")) digits)
                     (< decpt nd) (str (subs digits 0 decpt) "." (subs digits decpt))
                     :else (str digits (apply str (repeat (- decpt nd) "0")) ".0")))]
        (str (when minus? "-") body)))))

(defn- hex4 [n] (format "\\u%04x" n))

(defn- json-escape
  "json.dumps string escaping. ensure_ascii=False only escapes controls,
  backslash and quote; ensure_ascii=True also \\u-escapes every codepoint
  outside 0x20..0x7E (astral ones as a surrogate pair)."
  [s ensure-ascii?]
  (let [sb (StringBuilder.)]
    (doseq [c s]
      (let [cp (int c)]
        (cond
          (= c \") (.append sb "\\\"")
          (= c \\) (.append sb "\\\\")
          (= c \newline) (.append sb "\\n")
          (= c \return) (.append sb "\\r")
          (= c \tab) (.append sb "\\t")
          (= cp 8) (.append sb "\\b")
          (= cp 12) (.append sb "\\f")
          (< cp 0x20) (.append sb (hex4 cp))
          (and ensure-ascii? (> cp 0x7E))
          (if (< cp 0x10000)
            (.append sb (hex4 cp))
            (let [v (- cp 0x10000)]
              (.append sb (hex4 (bit-or 0xD800 (bit-shift-right v 10))))
              (.append sb (hex4 (bit-or 0xDC00 (bit-and v 0x3FF))))))
          :else (.append sb c))))
    (.toString sb)))

(defn- jkey [k] (if (keyword? k) (name k) (str k)))

(defn json-str
  "Serialize v like Python json.dumps(v) with the default separators
  (\", \" and \": \"). ensure_ascii defaults to FALSE here because the
  state is serialized that way (serialize_state); pass {:ensure-ascii true}
  for json.dumps' own default (_to_internal's instructions)."
  ([v] (json-str v {}))
  ([v {:keys [ensure-ascii] :as opts}]
   (cond
     (string? v) (str "\"" (json-escape v ensure-ascii) "\"")
     (keyword? v) (json-str (name v) opts)
     (map? v) (str "{" (str/join ", " (map (fn [[k val]]
                                              (str (json-str (jkey k) opts) ": " (json-str val opts)))
                                            v)) "}")
     (or (sequential? v) (set? v)) (str "[" (str/join ", " (map #(json-str % opts) v)) "]")
     (boolean? v) (if v "true" "false")
     (nil? v) "null"
     (integer? v) (str v)
     (number? v) (py-float-str v)
     :else (json-str (str v) opts))))

(defn serialize-state [state]
  (if (string? state) state (json-str state)))

(defn- falsy?
  "Python `not v`: None, False, 0/0.0, empty string and empty collections."
  [v]
  (or (nil? v) (false? v) (= v "")
      (and (number? v) (zero? v))
      (and (coll? v) (empty? v))))

(defn render-options
  "Option texts in label-index order. noul is always [false, true] so p[1] == noul."
  [q]
  (let [t (:t q) crit (:crit q)]
    (case t
      "choice" (mapv (fn [[k v]] (if (falsy? v) (str k) (str k ": " v))) crit)
      "score" (mapv (fn [i c] (str "level " i ": " c)) (range) crit)
      (let [c (or crit {})]
        [(str "false: " (or (get c "false") "no, the statement does not hold"))
         (str "true: " (or (get c "true") "yes, the statement holds"))]))))

(defn temp-bucket
  "Per-cardinality temperature key, e.g. \"choice:3-5\"."
  [qtype k]
  (let [size (if (<= k 2) "2" (if (<= k 5) "3-5" (if (<= k 10) "6-10" "11+")))]
    (str (qtype-names qtype) ":" size)))

(defn build-sequence
  "Returns [ids markers]: token ids and the [MASK] marker positions."
  [tok state q max-len head-max-len]
  (let [mask "[MASK]"
        sp (:specials tok)
        opts (render-options q)
        ins (str/replace (str (:ins q)) mask " ")
        raw-head (tk/encode tok (str (:t q) " question: " ins))
        opt-ids (mapv (fn [o]
                        (into [(:mask sp)]
                              (take 48 (tk/encode tok (str " " (str/replace o mask " "))))))
                      opts)
        opt-budget (- head-max-len (reduce + (map count opt-ids)))
        opt-ids (if (< opt-budget 16)
                  (let [per (max 4 (quot (- head-max-len 16) (max 1 (count opt-ids))))]
                    (mapv #(vec (take per %)) opt-ids))
                  opt-ids)
        opt-budget (- head-max-len (reduce + (map count opt-ids)))
        head-ids (vec (take (max 8 opt-budget) raw-head))
        base (into [(:cls sp)] (concat head-ids [(:sep sp)]))
        [ids markers] (reduce (fn [[ids ms] o]
                                [(into ids o) (conj ms (count ids))])
                              [base []]
                              opt-ids)
        ids (conj ids (:sep sp))
        room (max 0 (- max-len (count ids) 1))
        st (vec (take room (tk/encode tok (str/replace (serialize-state state) mask " "))))]
    [(vec (take max-len (into ids (conj st (:sep sp)))))
     (filterv #(< % max-len) markers)]))
