(ns laya.tokenizer-test
  "Tokenizer parity against the real transformers fast tokenizer.

  golden/tok.edn - input_ids for 12 tricky cases (whitespace runs, NFD
  folding, emoji/CJK, URLs, currency/scientific notation) captured from the
  checkpoint's own tokenizer. The pre-tokenizer must match the ordered
  alternation, and the added-token set (space runs, |||EMAIL_ADDRESS|||,
  ...) must be extracted leftmost-longest before ByteLevel pre-tokenization."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [laya.tokenizer :as tk]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LAYA_DATA") "data"))

(def tok (delay (tk/load (str data-dir "/tokenizer.edn"))))

(def cases
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
   "$1,234.56 —  -99.2e-3  +0.5"])

(deftest encode-matches-golden
  (let [golden (:cases (edn/read-string (slurp (str golden-dir "/tok.edn"))))]
    (doseq [i (range (count cases))]
      (testing (str "case " i)
        (is (= (get golden (str i))
               (tk/encode @tok (nth cases i))))))))

(deftest nfc-folds
  (is (= "café" (tk/nfc "cafe\u0301")))
  (is (= "ẹ́" (tk/nfc "e\u0301\u0323")))
  (is (= "plain" (tk/nfc "plain"))))
