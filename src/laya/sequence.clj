(ns laya.sequence
  "build_sequence + option rendering, mirroring rl_common.py.

  Sequence layout:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  The [MASK] positions (one per option, in label order) are returned as markers."
  (:require [clojure.string :as str]
            [laya.tokenizer :as tk]))

(def qtypes {"choice" 0 "score" 1 "noul" 2})
(def qtype-names {0 "choice" 1 "score" 2 "noul"})

(defn- json-escape [s]
  (let [sb (StringBuilder.)]
    (doseq [c s]
      (cond
        (= c \") (.append sb "\\\"")
        (= c \\) (.append sb "\\\\")
        (= c \newline) (.append sb "\\n")
        (= c \return) (.append sb "\\r")
        (= c \tab) (.append sb "\\t")
        (< (int c) 0x20) (.append sb (format "\\u%04x" (int c)))
        :else (.append sb c)))
    (.toString sb)))

(defn- jkey [k] (if (keyword? k) (name k) (str k)))

(defn json-str
  "Serialize v like Python json.dumps(v, ensure_ascii=False) for maps/vectors/
  strings/scalars (default separators \", \" and \": \")."
  [v]
  (cond
    (string? v) (str "\"" (json-escape v) "\"")
    (map? v) (str "{" (str/join ", " (map (fn [[k val]] (str (json-str (jkey k)) ": " (json-str val))) v)) "}")
    (vector? v) (str "[" (str/join ", " (map json-str v)) "]")
    (boolean? v) (if v "true" "false")
    (nil? v) "null"
    (number? v) (str v)
    :else (json-str (str v))))

(defn serialize-state [state]
  (if (string? state) state (json-str state)))

(defn- falsy? [v] (or (nil? v) (false? v) (= v "") (= v 0)))

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
