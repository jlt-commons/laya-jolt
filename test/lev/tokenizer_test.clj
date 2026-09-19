(ns lev.tokenizer-test
  "Tokenizer parity against the real transformers fast tokenizer.

  golden/tok.edn - input_ids for 12 tricky cases (whitespace runs, NFD
  folding, emoji/CJK, URLs, currency/scientific notation) captured from the
  checkpoint's own tokenizer. The pre-tokenizer must match the ordered
  alternation, and the added-token set (space runs, |||EMAIL_ADDRESS|||,
  ...) must be extracted leftmost-longest before ByteLevel pre-tokenization."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [lev.test-util :as tu :refer [golden-dir data-dir]]
            [lev.tokenizer :as tk]))

(def tok (delay (tk/load (str data-dir "/tokenizer.edn"))))

(def cases
  "The tokenizer cases from the golden dump (also listed in
  golden/cases.edn :tok-cases, which the tests check against)."
  ["Which department should handle this email?"
   " choice question: invoices, payments, refunds"
   "  leading double space and\ttab"
   "Don't stop — it's 2024, isn't it? naïve café résumé ﬁne"
   "unicode\u0301 combining acute (NFD source; NFC must fold)"
   "MixedCASE WithNumbers 123 45.67 and #hashtags @at"
   "level 0: not urgent"
   "true: yes, the statement holds"
   "Duplicate billing on March invoice #4411"
   "https://example.com/path?q=1&r=2&x=%20"
   "emoji 😀 and cjk 中文测试"
   "$1,234.56 —  -99.2e-3  +0.5"
   "and\tThe end"
   "line1\nline2"
   "a\u001cb\u001fc"
   "x\u00a0y"
   "tab\t\tdouble"
   "nl\n\n\nrun"
   "a \tb"
   "DON'T Shout 'S"
   "end   "
   "\u00a0\u00a0lead"
   "\u2028sep"
   "'re're 're"
   "z\n\n\n\n\nq"
   "\r\nThe"])

(deftest cases-match-oracle-inputs
  (is (= cases (:tok-cases (tu/read-golden "cases")))
      "the local case list drifted from the one the oracle encoded"))

(deftest encode-matches-golden
  (let [golden (:cases (edn/read-string (slurp (str golden-dir "/tok.edn"))))]
    (doseq [i (range (count cases))]
      (testing (str "case " i)
        (is (= (get golden (str i))
               (tk/encode @tok (nth cases i))))))))

(deftest pre-tokenize-matches-hf-scanner
  (testing "hand scanner == tokenizers ByteLevel(use_regex=true) boundaries"
    (let [golden (:pre-tokens (tu/read-golden "cases"))]
      (doseq [c cases]
        (is (= (get golden c) (tk/pre-tokenize c)) (pr-str c)))))
  (testing "only U+0020 is the optional prefix; other whitespace stands alone"
    (is (= ["and" "\t" "The" " end"] (tk/pre-tokenize "and\tThe end")))
    (is (= ["x" "\u00a0" "y"] (tk/pre-tokenize "x\u00a0y"))))
  (testing "\\s is Unicode White_Space: U+001C..1F are 'other', not space"
    (is (= ["a" "\u001c" "b"] (tk/pre-tokenize "a\u001cb"))))
  (testing "a whitespace run before a word gives up its last char"
    (is (= ["ab" "\n\n\n\n" "\n" "q"] (tk/pre-tokenize "ab\n\n\n\n\nq")))
    (is (= ["ab" "  " " q"] (tk/pre-tokenize "ab   q")))
    (is (= ["ab" "   "] (tk/pre-tokenize "ab   ")))))

(deftest nfc-folds
  (is (= "café" (tk/nfc "cafe\u0301")))
  (is (= "ẹ́" (tk/nfc "e\u0301\u0323")))
  (is (= "plain" (tk/nfc "plain")))
  (testing "output may be longer than the input (composition exclusions)"
    ;; U+0958 decomposes to U+0915 U+093C and is excluded from recomposition,
    ;; so NFC doubles the length; the old n+32 buffer overflowed here.
    (let [n 300]
      (is (= (apply str (repeat n "\u0915\u093c"))
             (tk/nfc (apply str (repeat n "\u0958"))))))))

;; --- the sentencepiece kind (mmBERT / Gemma): unit tests for its stages ----------

(deftest metaspace-pieces
  (testing "spaces become ▁, a ▁ is prepended, the split keeps ▁ with what follows"
    (is (= ["▁Hello" "▁world"] (tk/metaspace-pieces "Hello world")))
    (is (= ["▁Hello" "▁world"] (tk/metaspace-pieces " Hello world")))
    (is (= ["▁a" "▁" "▁b"] (tk/metaspace-pieces "a  b")))
    (is (= ["▁end" "▁" "▁" "▁"] (tk/metaspace-pieces "end   ")))
    (is (= ["▁already" "▁marked"] (tk/metaspace-pieces "▁already marked")) "a literal ▁ is not doubled")
    (is (= ["▁tab\tsep"] (tk/metaspace-pieces "tab\tsep")) "only U+0020 is a separator")
    (is (= ["▁x y"] (tk/metaspace-pieces "x y")))
    (is (= ["▁"] (tk/metaspace-pieces " ")))
    (is (= [] (tk/metaspace-pieces "")) "an empty segment yields nothing, not ▁")))

(deftest sentencepiece-symbols
  (let [vocab {"▁" 1 "a" 2 "<0xF0>" 10 "<0x90>" 11 "<0x8D>" 12 "<0x88>" 13 "<unk>" 3 "▁a" 4}]
    (testing "chars in the vocab stay chars; others fall back to their UTF-8 bytes as <0xNN>"
      (is (= ["▁" "a"] (tk/sp-symbols "▁a" vocab true true "<unk>")))
      (is (= ["▁" "<0xF0>" "<0x90>" "<0x8D>" "<0x88>"] (tk/sp-symbols "▁𐍈" vocab true true "<unk>"))
          "U+10348 is four bytes"))
    (testing "without byte fallback, or a missing byte token, it is <unk>; consecutive unks fuse"
      (is (= ["▁" "<unk>"] (tk/sp-symbols "▁𐍈" vocab false true "<unk>")))
      (is (= ["▁" "<unk>"] (tk/sp-symbols "▁zz" vocab false true "<unk>")))
      (is (= ["▁" "<unk>" "<unk>"] (tk/sp-symbols "▁zz" vocab false false "<unk>"))))))

(deftest added-tokens-with-lstrip
  (let [added {"<mask>" 4 "<bos>" 2 "\n" 108 "\n\n" 109}]
    (testing "leftmost-longest, plain"
      (is (= ["a" 108 "b"] (tk/sp-segments "a\nb" added 6 #{})))
      (is (= ["a" 109 "b"] (tk/sp-segments "a\n\nb" added 6 #{})))
      (is (= ["text with " 4 " inside"] (tk/sp-segments "text with <mask> inside" added 6 #{}))))
    (testing "lstrip swallows the whitespace before the token, but not a previous match"
      (is (= ["text with" 4 " inside"] (tk/sp-segments "text with <mask> inside" added 6 #{"<mask>"})))
      (is (= ["a" 108 4] (tk/sp-segments "a\n<mask>" added 6 #{"<mask>"})))
      (is (= ["a" 4] (tk/sp-segments "a \t <mask>" added 6 #{"<mask>"})) "any Unicode White_Space"))))
