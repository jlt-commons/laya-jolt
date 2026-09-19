(ns lev.json
  "A small strict JSON reader (RFC 8259) whose objects keep their key order.

  The HTTP API's JSON is model input: the order of a choice's options, of
  the questions, and of a state object's fields all change the token
  sequence, and Python's json.loads preserves it. clojure.data.json (used by
  prepare, where order is irrelevant) builds hash-maps and loses it past 8
  keys, so requests are read with this one.

  Objects become insertion-ordered maps, arrays vectors. Numbers follow
  Python's json module: an integer literal reads as an integer, anything
  with a fraction or exponent as a double."
  (:require [clojure.edn :as edn]
            [lev.sequence :as seq]))

(defn- fail [s i msg]
  (throw (ex-info (str "json: " msg " at offset " i)
                  {:type :invalid-json :offset i
                   :near (subs s i (min (count s) (+ i 20)))})))

(defn- skip-ws [^String s i n]
  (loop [i i]
    (if (and (< i n) (let [c (.charAt s i)] (or (= c \space) (= c \tab) (= c \newline) (= c \return))))
      (recur (inc i))
      i)))

(declare read-value)

(defn- hex4 [^String s i n]
  (when (> (+ i 4) n) (fail s i "truncated \\u escape"))
  (let [h (subs s i (+ i 4))]
    (when-not (re-matches #"[0-9a-fA-F]{4}" h) (fail s i "bad \\u escape"))
    (Long/parseLong h 16)))

(defn- read-string* [^String s i n]
  ;; i points just past the opening quote
  (let [sb (StringBuilder.)]
    (loop [i i]
      (when (>= i n) (fail s i "unterminated string"))
      (let [c (.charAt s i)]
        (cond
          (= c \") [(.toString sb) (inc i)]
          (= c \\)
          (do (when (>= (inc i) n) (fail s i "truncated escape"))
              (let [e (.charAt s (inc i))]
                (case e
                  \" (do (.append sb \") (recur (+ i 2)))
                  \\ (do (.append sb \\) (recur (+ i 2)))
                  \/ (do (.append sb \/) (recur (+ i 2)))
                  \b (do (.append sb (char 8)) (recur (+ i 2)))
                  \f (do (.append sb (char 12)) (recur (+ i 2)))
                  \n (do (.append sb \newline) (recur (+ i 2)))
                  \r (do (.append sb \return) (recur (+ i 2)))
                  \t (do (.append sb \tab) (recur (+ i 2)))
                  \u (let [u (hex4 s (+ i 2) n)]
                       ;; a high surrogate followed by \\uDCxx is one astral char
                       (if (and (>= u 0xD800) (< u 0xDC00)
                                (<= (+ i 12) n)
                                (= \\ (.charAt s (+ i 6))) (= \u (.charAt s (+ i 7))))
                         (let [lo (hex4 s (+ i 8) n)]
                           (if (and (>= lo 0xDC00) (< lo 0xE000))
                             (do (.append sb (char (+ 0x10000
                                                     (bit-shift-left (- u 0xD800) 10)
                                                     (- lo 0xDC00))))
                                 (recur (+ i 12)))
                             (fail s i "lone surrogate")))
                         (if (and (>= u 0xD800) (< u 0xE000))
                           (fail s i "lone surrogate")
                           (do (.append sb (char u)) (recur (+ i 6))))))
                  (fail s i (str "bad escape \\" e)))))
          (< (int c) 0x20) (fail s i "control character in string")
          :else (do (.append sb c) (recur (inc i))))))))

(defn- number-char? [c]
  (or (and (>= (int c) 48) (<= (int c) 57))
      (= c \-) (= c \+) (= c \.) (= c \e) (= c \E)))

(def ^:private number-re #"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?")

(defn- read-number [^String s i n]
  (let [j (loop [j i] (if (and (< j n) (number-char? (.charAt s j))) (recur (inc j)) j))
        lit (subs s i j)]
    (when-not (re-matches number-re lit) (fail s i (str "bad number " (pr-str lit))))
    [(if (re-find #"[.eE]" lit)
       (Double/parseDouble lit)
       (edn/read-string lit))   ; exact integer, bignum past 64 bits
     j]))

(defn- read-array [^String s i n]
  (let [i (skip-ws s i n)]
    (if (and (< i n) (= \] (.charAt s i)))
      [[] (inc i)]
      (loop [i i out (transient [])]
        (let [[v i] (read-value s i n)
              out (conj! out v)
              i (skip-ws s i n)]
          (when (>= i n) (fail s i "unterminated array"))
          (case (.charAt s i)
            \, (recur (skip-ws s (inc i) n) out)
            \] [(persistent! out) (inc i)]
            (fail s i "expected , or ]")))))))

(defn- read-object [^String s i n]
  (let [i (skip-ws s i n)]
    (if (and (< i n) (= \} (.charAt s i)))
      [(array-map) (inc i)]
      (loop [i i kvs (transient [])]
        (when (or (>= i n) (not= \" (.charAt s i))) (fail s i "expected a string key"))
        (let [[k i] (read-string* s (inc i) n)
              i (skip-ws s i n)]
          (when (or (>= i n) (not= \: (.charAt s i))) (fail s i "expected :"))
          (let [[v i] (read-value s (skip-ws s (inc i) n) n)
                kvs (conj! kvs [k v])
                i (skip-ws s i n)]
            (when (>= i n) (fail s i "unterminated object"))
            (case (.charAt s i)
              \, (recur (skip-ws s (inc i) n) kvs)
              \} [(seq/ordered-map (persistent! kvs)) (inc i)]
              (fail s i "expected , or }"))))))))

(defn- read-literal [^String s i n word value]
  (let [j (+ i (count word))]
    (if (and (<= j n) (= word (subs s i j)))
      [value j]
      (fail s i (str "expected " word)))))

(defn- read-value [^String s i n]
  (when (>= i n) (fail s i "unexpected end of input"))
  (let [c (.charAt s i)]
    (cond
      (= c \{) (read-object s (inc i) n)
      (= c \[) (read-array s (inc i) n)
      (= c \") (read-string* s (inc i) n)
      (= c \t) (read-literal s i n "true" true)
      (= c \f) (read-literal s i n "false" false)
      (= c \n) (read-literal s i n "null" nil)
      (or (= c \-) (and (>= (int c) 48) (<= (int c) 57))) (read-number s i n)
      :else (fail s i (str "unexpected character " (pr-str c))))))

(defn read-str
  "Parse one JSON document from s. Trailing non-whitespace is an error.
  Throws ex-info {:type :invalid-json :offset ...}."
  [^String s]
  (let [n (count s)
        [v i] (read-value s (skip-ws s 0 n) n)
        i (skip-ws s i n)]
    (when (< i n) (fail s i "trailing content"))
    v))
