(ns laya.tokenizer
  "GPT-2 byte-level BPE, faithful to the checkpoint's tokenizer.json:
  NFC normalize (ICU unorm2 via FFI), GPT-2 pre-tokenization (hand-written
  scanner with exact \\p{L}/\\p{N} classes via ICU u_charType, since the
  pattern needs lookahead that irregex lacks), byte-to-unicode mapping, and
  lowest-rank-first BPE merges.

  Encoding parity is pinned by golden/tok.edn (input_ids from the real
  transformers fast tokenizer on 12 tricky cases)."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [jolt.ffi :as ffi]))

;; --- ICU: NFC + character classes ---------------------------------------------

(ffi/defcfn unorm2-get-instance* "unorm2_getNFCInstance" [] :pointer)
(ffi/defcfn unorm2-normalize* "unorm2_normalize"
  [:pointer :pointer :int32 :pointer :int32 :pointer] :int32)
(ffi/defcfn u-char-type* "u_charType" [:int32] :int32)
(ffi/defcfn u-isspace* "u_isspace" [:int32] :int32)

(def nfc-instance (delay (unorm2-get-instance*)))

(defn utf8->u16
  "Codepoint string -> vector of UTF-16 code units."
  [s]
  (loop [chs (seq s) out (transient [])]
    (if (empty? chs)
      (persistent! out)
      (let [cp (long (first chs))]
        (if (< cp 0x10000)
          (recur (next chs) (conj! out cp))
          (let [v (- cp 0x10000)]
            (recur (next chs)
                   (-> out
                       (conj! (bit-or 0xD800 (bit-shift-right v 10)))
                       (conj! (bit-or 0xDC00 (bit-and v 0x3FF)))))))))))

(defn u16->string
  "Vector of UTF-16 code units -> codepoint string (jolt strings are
  codepoint-indexed, so surrogates collapse to single chars)."
  [units]
  (let [cps (loop [us (seq units) out ()]
              (if (empty? us)
                (reverse out)
                (let [u (first us)]
                  (if (and (>= u 0xD800) (< u 0xDC00)
                           (seq (rest us))
                           (>= (second us) 0xDC00) (< (second us) 0xE000))
                    (recur (rest (rest us))
                           (conj out (+ 0x10000
                                        (bit-shift-left (bit-and (- u 0xD800) 0x3FF) 10)
                                        (bit-and (- (second us) 0xDC00) 0x3FF))))
                    (recur (rest us) (conj out u))))))]
    (apply str (map char cps))))

(defn nfc
  "NFC-normalize via ICU. ASCII-only input is identity (fast path)."
  [s]
  (if (every? #(< (long %) 128) s)
    s
    (let [units (utf8->u16 s)
          n (count units)
          src (ffi/alloc (* 2 (max n 1)))
          dst (ffi/alloc (* 2 (+ n 32)))
          err (ffi/alloc 4)]
      (dotimes [i n]
        (ffi/write src :uint16 (bit-and 0xFFFF (nth units i)) (* 2 i)))
      (let [ret (unorm2-normalize* @nfc-instance src (int n)
                                   dst (int (+ n 32)) err)
            code (ffi/read err :int32 0)]
        (when (not (zero? code))
          (throw (ex-info "unorm2_normalize failed" {:code code})))
        (u16->string (mapv #(ffi/read dst :uint16 (* 2 %)) (range ret)))))))

;; \\p{L} = UCharType 1..5; \\p{N} = 9..11 (Nd, Nl, No)
(defn letter? [c] (let [t (u-char-type* (int c))] (and (>= t 1) (<= t 5))))
(defn number? [c] (let [t (u-char-type* (int c))] (and (>= t 9) (<= t 11))))
(defn space? [c] (not (zero? (u-isspace* (int c)))))

;; --- byte-level BPE -----------------------------------------------------------

(def ^:private byte-chars
  "GPT-2 bytes_to_unicode: bytes in the printable ranges map to themselves,
  every other byte b maps to 256+n (n = its ordinal among excluded bytes)."
  (let [in-ranges (fn [b]
                    (boolean (some (fn [[lo hi]] (and (>= b lo) (<= b hi)))
                                   [[33 126] [161 172] [174 255]])))]
    (loop [b 0 n 0 out []]
      (if (= b 256)
        out
        (if (in-ranges b)
          (recur (inc b) n (conj out b))
          (recur (inc b) (inc n) (conj out (+ 256 n))))))))

(defn string->utf8-bytes
  "Codepoint string -> UTF-8 byte sequence."
  [s]
  (mapcat (fn [ch]
            (let [cp (long ch)]
              (cond
                (< cp 0x80) [cp]
                (< cp 0x800) [(bit-or 0xC0 (bit-shift-right cp 6))
                              (bit-or 0x80 (bit-and cp 0x3F))]
                (< cp 0x10000) [(bit-or 0xE0 (bit-shift-right cp 12))
                                (bit-or 0x80 (bit-and (bit-shift-right cp 6) 0x3F))
                                (bit-or 0x80 (bit-and cp 0x3F))]
                :else [(bit-or 0xF0 (bit-shift-right cp 18))
                       (bit-or 0x80 (bit-and (bit-shift-right cp 12) 0x3F))
                       (bit-or 0x80 (bit-and (bit-shift-right cp 6) 0x3F))
                       (bit-or 0x80 (bit-and cp 0x3F))])))
          s))

(defn piece->chars
  "One pre-token -> the vocab-key char sequence (byte-level encoding)."
  [s]
  (map #(char (nth byte-chars %)) (string->utf8-bytes s)))

(defn pre-tokenize
  "GPT-2 pattern, ordered alternation:
    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
  Hand scanner over codepoints; returns pre-token strings."
  [s]
  (let [cps (vec s)
        n (count cps)
        contraction? (fn [i]
                       (when (and (= (nth cps i) \') (< (inc i) n))
                         (let [c1 (nth cps (inc i))
                               c2 (when (< (+ i 2) n) (nth cps (+ i 2)))]
                           (cond (and (= c1 \s) (letter? c1)) (+ i 2)
                                 (= c1 \t) (+ i 2)
                                 (and (= c1 \r) (= c2 \e)) (+ i 3)
                                 (and (= c1 \v) (= c2 \e)) (+ i 3)
                                 (and (= c1 \m)) (+ i 2)
                                 (and (= c1 \l) (= c2 \l)) (+ i 3)
                                 (= c1 \d) (+ i 2)
                                 :else nil))))
        run (fn [i pred]
              (loop [k i]
                (if (or (>= k n) (not (pred (nth cps k))))
                  k
                  (recur (inc k)))))
        other-run (fn [i]
                    (loop [k i]
                      (if (or (>= k n)
                              (space? (nth cps k))
                              (letter? (nth cps k))
                              (number? (nth cps k)))
                        k
                        (recur (inc k)))))]
    (loop [i 0 rev ()]
      (if (>= i n)
        (vec (reverse rev))
        (let [c (nth cps i)
              end (or (contraction? i)
                      (cond
                        (and (space? c) (< (inc i) n) (letter? (nth cps (inc i))))
                        (run (inc i) letter?)

                        (letter? c) (run i letter?)

                        (and (space? c) (< (inc i) n) (number? (nth cps (inc i))))
                        (run (inc i) number?)

                        (number? c) (run i number?)

                        (and (space? c) (< (inc i) n)
                             (not (space? (nth cps (inc i)))))
                        (other-run (inc i))

                        (space? c)
                        (let [k (run i space?)]
                          (if (< k n) (dec k) k))

                        :else (other-run i)))
              ;; a space followed by a token starts the token; i stays the
              ;; token start in every branch above
              tok (subs s i end)]
          (recur end (conj rev tok)))))))

(defrecord Tokenizer [vocab ranks specials added max-added])

(defn load
  "Load data/tokenizer.edn -> Tokenizer."
  [path]
  (let [m (edn/read-string (slurp path))
        added (into {} (:added m))]
    (->Tokenizer (:vocab m)
                 (into {} (map-indexed (fn [i mg]
                                         (let [[a b] (str/split mg #" ")]
                                           [[a b] i])))
                       (:merges m))
                 (:specials m)
                 added
                 (reduce (fn [mx [k _]] (max mx (count k))) 0 added))))

(defn apply-bpe
  "GPT-2 BPE: repeatedly merge the adjacent pair with the lowest rank."
  [parts ranks]
  (loop [parts parts]
    (if (< (count parts) 2)
      parts
      (let [best (reduce (fn [best [a b]]
                           (let [r (get ranks [a b])]
                             (if (and r (or (nil? best) (< r (best 2))))
                               [a b r]
                               best)))
                         nil
                         (map vector parts (rest parts)))]
        (if (nil? best)
          parts
          (let [[a b] best
                merged (str a b)]
            (recur (loop [src parts out []]
                     (if (empty? src)
                       out
                       (if (and (= (first src) a)
                                (seq (rest src))
                                (= (second src) b))
                         (recur (rest (rest src)) (conj out merged))
                         (recur (rest src) (conj out (first src)))))))))))))

(defn added-segments
  "Split s leftmost-longest on the added-token set. Returns a vector of
  strings (gaps to pre-tokenize) and longs (added-token ids to emit verbatim).
  HuggingFace extracts added tokens after normalization and before the
  ByteLevel pre-tokenizer, so a run like \"  \" becomes its own id rather
  than a pair of byte-level spaces."
  [s added max-added]
  (let [n (count s)]
    (loop [i 0 start 0 out []]
      (if (>= i n)
        (if (< start n) (conj out (subs s start n)) out)
        (let [lim (min max-added (- n i))
              hit (loop [l lim]
                    (when (pos? l)
                      (let [sub (subs s i (+ i l))]
                        (if (contains? added sub) l (recur (dec l))))))]
          (if hit
            (recur (+ i hit) (+ i hit)
                   (conj (if (< start i) (conj out (subs s start i)) out)
                         (get added (subs s i (+ i hit)))))
            (recur (inc i) start out)))))))

(defn encode
  "Text -> input_ids (no special tokens), matching
  tok(text, add_special_tokens=False)['input_ids']."
  [tok text]
  (let [s (nfc text)]
    (into []
          (mapcat (fn [seg]
                    (if (string? seg)
                      (mapcat (fn [pre]
                                (let [parts (apply-bpe (map str (piece->chars pre))
                                                       (:ranks tok))]
                                  (keep #(get (:vocab tok) %) parts)))
                              (pre-tokenize seg))
                      [seg])))
          (added-segments s (:added tok) (:max-added tok)))))

(defn special
  "id of :cls/:sep/:pad/:mask/:unk."
  [tok k]
  (get (:specials tok) k))
