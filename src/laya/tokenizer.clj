(ns laya.tokenizer
  "The two tokenizers in the Laya bundle, faithful to their tokenizer.json:

  :byte-level (english, typed-decisions) - GPT-2 byte-level BPE: NFC
  normalize (ICU unorm2 via FFI), GPT-2 pre-tokenization (hand-written
  scanner with exact \\p{L}/\\p{N} classes via ICU u_charType, since the
  pattern needs lookahead that irregex lacks), byte-to-unicode mapping, and
  lowest-rank-first BPE merges.

  :sentencepiece (multilingual, the mmBERT/Gemma tokenizer) - added tokens
  extracted first (leftmost-longest, `lstrip` swallowing whitespace), then
  per segment: spaces to \u2581, a \u2581 prepended, split keeping \u2581
  with the piece it starts, and BPE over each piece with UTF-8 byte
  fallback (<0xNN>) and fused <unk>.

  Encoding parity is pinned by golden/tok.edn and golden/multilingual/tok.edn
  (input_ids from the real transformers fast tokenizers) and the scanner by
  the ByteLevel pre-tokenizer boundaries in golden/cases.edn."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [jolt.ffi :as ffi]))

;; --- ICU: NFC + character classes ---------------------------------------------

(defmacro ^:private deficu
  "defcfn for an ICU symbol. mac's libicucore exports plain names; linux ICU
  builds append the major version (u_charType_76), so resolve the first name
  that exists at expansion time. Extend the version list if your ICU is newer."
  [name csym argtypes rettype]
  (let [candidates (cons csym (map #(str csym "_" %) (range 90 59 -1)))
        found (some #(when (ffi/find-symbol %) %) candidates)]
    (when-not found
      (throw (ex-info (str "ICU symbol not found: " csym
                           " (is libicuuc / libicucore declared in :jolt/native?)")
                      {:symbol csym})))
    `(def ~name (ffi/foreign-fn ~found ~argtypes ~rettype))))

;; takes UErrorCode* and reads it before doing anything: passing nothing
;; "worked" on mac by luck and segfaulted on linux
(deficu unorm2-get-instance* "unorm2_getNFCInstance" [:pointer] :pointer)
(deficu unorm2-normalize* "unorm2_normalize"
  [:pointer :pointer :int32 :pointer :int32 :pointer] :int32)
(deficu u-char-type* "u_charType" [:int32] :int32)
;; White_Space property == regex \\s. NOT u_isspace, which also admits the
;; U+001C..U+001F separators that the GPT-2 pattern treats as ordinary chars.
(deficu u-is-uwhitespace* "u_isUWhiteSpace" [:int32] :int8)

(def nfc-instance
  (delay (with-open [a (ffi/confined-arena)]
           (let [err (ffi/alloc a 4)]
             (ffi/write err :int32 0 0)
             (let [p (unorm2-get-instance* err)
                   code (ffi/read err :int32 0)]
               (when (pos? code)
                 (throw (ex-info "unorm2_getNFCInstance failed" {:code code})))
               p)))))

(def ^:private U_BUFFER_OVERFLOW_ERROR 15)

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
          ;; NFC can grow a string (composition exclusions decompose without
          ;; recomposing), by at most 3x. Start there; retry on overflow with
          ;; the length ICU reports, so no input can fail on buffer size.
          normalize (fn [cap]
                      (with-open [a (ffi/confined-arena)]
                        (let [src (ffi/alloc a (* 2 (max n 1)))
                              dst (ffi/alloc a (* 2 cap))
                              err (ffi/alloc a 4)]
                          (dotimes [i n]
                            (ffi/write src :uint16 (bit-and 0xFFFF (nth units i)) (* 2 i)))
                          ;; ICU returns early on a pre-set error code
                          (ffi/write err :int32 0 0)
                          (let [ret (unorm2-normalize* @nfc-instance src (int n)
                                                       dst (int cap) err)
                                code (ffi/read err :int32 0)]
                            (cond
                              (zero? code)
                              (u16->string (mapv #(ffi/read dst :uint16 (* 2 %)) (range ret)))
                              (= code U_BUFFER_OVERFLOW_ERROR) [:retry ret]
                              :else (throw (ex-info "unorm2_normalize failed" {:code code})))))))
          r (normalize (+ (* 3 n) 16))]
      (if (vector? r) (normalize (inc (second r))) r))))

;; \\p{L} = UCharType 1..5; \\p{N} = 9..11 (Nd, Nl, No)
(defn letter? [c] (let [t (u-char-type* (int c))] (and (>= t 1) (<= t 5))))
(defn number? [c] (let [t (u-char-type* (int c))] (and (>= t 9) (<= t 11))))
(defn space? [c] (not (zero? (u-is-uwhitespace* (int c)))))

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
  Hand scanner over codepoints; returns pre-token strings.

  The optional prefix is a literal U+0020 only: a tab or NBSP before a word
  is its own \\s+ token. A whitespace run followed by a non-space gives up
  its last char (\\s+(?!\\S) backtracks one), unless the run is a single
  char, in which case plain \\s+ takes it."
  [s]
  (let [cps (vec s)
        n (count cps)
        contraction? (fn [i]
                       (when (and (= (nth cps i) \') (< (inc i) n))
                         (let [c1 (nth cps (inc i))
                               c2 (when (< (+ i 2) n) (nth cps (+ i 2)))]
                           (cond (= c1 \s) (+ i 2)
                                 (= c1 \t) (+ i 2)
                                 (and (= c1 \r) (= c2 \e)) (+ i 3)
                                 (and (= c1 \v) (= c2 \e)) (+ i 3)
                                 (= c1 \m) (+ i 2)
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
                        (recur (inc k)))))
        next-c (fn [i] (when (< (inc i) n) (nth cps (inc i))))]
    (loop [i 0 rev ()]
      (if (>= i n)
        (vec (reverse rev))
        (let [c (nth cps i)
              nc (next-c i)
              end (or (contraction? i)
                      (cond
                        ;; " ?\\p{L}+" / " ?\\p{N}+" / " ?[^\\s\\p{L}\\p{N}]+"
                        (and (= c \space) nc (letter? nc)) (run (inc i) letter?)
                        (letter? c) (run i letter?)
                        (and (= c \space) nc (number? nc)) (run (inc i) number?)
                        (number? c) (run i number?)
                        (and (= c \space) nc (not (space? nc))) (other-run (inc i))

                        ;; "\\s+(?!\\S)|\\s+"
                        (space? c)
                        (let [k (run i space?)]
                          (if (and (< k n) (> (- k i) 1)) (dec k) k))

                        :else (other-run i)))
              tok (subs s i end)]
          (recur end (conj rev tok)))))))

(defrecord Tokenizer [vocab ranks specials added max-added kind mask-token lstrip byte-fallback fuse-unk])

(defn load
  "Load data/tokenizer.edn -> Tokenizer. :kind defaults to :byte-level (the
  format-1 files jolt prepare writes for GPT-2 style tokenizers)."
  [path]
  (let [m (edn/read-string (slurp path))
        added (into {} (:added m))]
    (map->Tokenizer
     {:vocab (:vocab m)
      :ranks (into {} (map-indexed (fn [i mg]
                                     (let [[a b] (str/split mg #" ")]
                                       [[a b] i])))
                   (:merges m))
      :specials (:specials m)
      :added added
      :max-added (reduce (fn [mx [k _]] (max mx (count k))) 0 added)
      :kind (or (:kind m) :byte-level)
      :mask-token (or (:mask-token m) "[MASK]")
      :lstrip (set (:lstrip m))
      :byte-fallback (boolean (:byte-fallback m))
      :fuse-unk (boolean (:fuse-unk m))})))

(defn mask-token
  "The mask token's text ([MASK] or <mask>): build-sequence blanks it out of
  user text so it cannot inject a marker."
  [tok]
  (:mask-token tok))

(defn- lowest-ranked-pair
  "[a b rank] of the adjacent pair with the lowest merge rank, or nil.
  A reduce whose accumulator starts as nil and is tested with nil?: jolt
  0.8.9's release build typed the accumulator from its init alone and
  folded the nil? to true (the last pair won and every BPE merge in the
  standalone binary went wrong); 0.8.10 fixed it. `jolt binary` runs the
  built server's self-test against golden/ so a regression cannot ship."
  [parts ranks]
  (reduce (fn [best [a b]]
            (let [r (get ranks [a b])]
              (if (and r (or (nil? best) (< (long r) (long (nth best 2)))))
                [a b r]
                best)))
          nil
          (map vector parts (rest parts))))

(defn apply-bpe
  "GPT-2 BPE: repeatedly merge the adjacent pair with the lowest rank."
  [parts ranks]
  (loop [parts parts]
    (if (< (count parts) 2)
      parts
      (let [best (lowest-ranked-pair parts ranks)]
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

(defn- byte-level-encode [tok text]
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

;; --- sentencepiece BPE (mmBERT / Gemma) -----------------------------------------

(def ^:private lower-one-eighth-block "\u2581")

(defn sp-segments
  "Added-token extraction for the sentencepiece kind: like added-segments
  (leftmost-longest over the raw text; HF matches these before any
  normalization) but a token in `lstrip` also swallows the Unicode
  whitespace before it, up to the previous match. Strings are gaps, longs
  are ids."
  [s added max-added lstrip]
  (let [n (count s)
        emit-gap (fn [out from to] (if (< from to) (conj out (subs s from to)) out))]
    (loop [i 0 start 0 out []]
      (if (>= i n)
        (emit-gap out start n)
        (let [lim (min max-added (- n i))
              hit (loop [l lim]
                    (when (pos? l)
                      (let [sub (subs s i (+ i l))]
                        (if (contains? added sub) l (recur (dec l))))))]
          (if hit
            (let [content (subs s i (+ i hit))
                  from (if (contains? lstrip content)
                         (loop [k i] (if (and (> k start) (space? (nth s (dec k)))) (recur (dec k)) k))
                         i)]
              (recur (+ i hit) (+ i hit)
                     (conj (emit-gap out start from) (get added content))))
            (recur (inc i) start out)))))))

(defn metaspace-pieces
  "One gap -> BPE words, as tokenizers' Metaspace(prepend_scheme=always,
  split=true) after the Replace(\" \" -> \u2581) normalizer: every space is
  a \u2581, one more is prepended unless the text already starts with one,
  and the text is split so each piece starts with its \u2581 (an empty gap
  yields nothing: prepend on an empty string is a no-op)."
  [seg]
  (let [s (str/replace seg " " lower-one-eighth-block)]
    (if (empty? s)
      []
      (let [s (if (str/starts-with? s lower-one-eighth-block) s (str lower-one-eighth-block s))
            n (count s)]
        (loop [i 1 from 0 out []]
          (cond
            (>= i n) (conj out (subs s from n))
            (= (nth s i) (first lower-one-eighth-block)) (recur (inc i) i (conj out (subs s from i)))
            :else (recur (inc i) from out)))))))

(defn sp-symbols
  "BPE::merge_word's starting symbols for one piece: each char that is a
  vocab token as itself; otherwise its UTF-8 bytes as <0xNN> tokens when
  byte fallback is on and every one exists; otherwise the unk token, with
  consecutive unks fused into one when fuse-unk is on."
  [piece vocab byte-fallback? fuse-unk? unk]
  (loop [cs (seq piece) out [] unk-run? false]
    (if (empty? cs)
      out
      (let [c (str (first cs))]
        (if (contains? vocab c)
          (recur (rest cs) (conj out c) false)
          (let [bytes (when byte-fallback?
                        (let [ts (map #(format "<0x%02X>" %) (string->utf8-bytes c))]
                          (when (every? #(contains? vocab %) ts) ts)))]
            (cond
              bytes (recur (rest cs) (into out bytes) false)
              (and unk-run? fuse-unk?) (recur (rest cs) out true)
              :else (recur (rest cs) (conj out unk) true))))))))

(defn- sentencepiece-encode [tok text]
  (let [vocab (:vocab tok)
        unk (some (fn [[t i]] (when (= i (:unk (:specials tok))) t)) (:added tok))]
    (into []
          (mapcat (fn [seg]
                    (if (string? seg)
                      (mapcat (fn [piece]
                                (->> (sp-symbols piece vocab (:byte-fallback tok) (:fuse-unk tok) unk)
                                     (#(apply-bpe % (:ranks tok)))
                                     (keep #(get vocab %))))
                              (metaspace-pieces seg))
                      [seg])))
          (sp-segments text (:added tok) (:max-added tok) (:lstrip tok)))))

(defn encode
  "Text -> input_ids (no special tokens), matching
  tok(text, add_special_tokens=False)['input_ids'] for the tokenizer's kind."
  [tok text]
  (case (:kind tok)
    :sentencepiece (sentencepiece-encode tok text)
    (byte-level-encode tok text)))

(defn special
  "id of :cls/:sep/:pad/:mask/:unk."
  [tok k]
  (get (:specials tok) k))
