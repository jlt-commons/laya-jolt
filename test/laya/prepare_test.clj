(ns laya.prepare-test
  "jolt prepare must reproduce the reference Python conversion byte for
  byte: golden/prepare.edn pins the size and zlib CRC-32 of every
  file the Python converter wrote for this checkpoint. Needs the checkpoint
  (LAYA_HOME, default ../laya); writes to target/prepare-test and removes it."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [laya.prepare :as prep]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def laya-home
  (or (System/getenv "LAYA_HOME") "../laya"))

(deftest jolt-prepare-reproduces-python-conversion
  (is (.exists (io/file laya-home "model.safetensors"))
      (str "checkpoint not found under " laya-home " (set LAYA_HOME)"))
  (let [golden (:files (edn/read-string (slurp (str golden-dir "/prepare.edn"))))
        out "target/prepare-test"]
    (jolt.host/delete-tree! out)
    (prep/convert laya-home out)
    (testing "every file matches the oracle's size and crc32"
      (is (= 209 (count golden)))
      (doseq [[fn {:keys [size crc32]}] golden]
        (is (= [size crc32] (prep/file-crc32 (str out "/" fn))) fn)))
    (testing "nothing extra was written"
      (is (= (set (keys golden))
             (set (for [f (file-seq (io/file out)) :when (.isFile f)]
                    (subs (.getPath f) (inc (count out))))))))
    (jolt.host/delete-tree! out)))

(deftest edn-writer-escapes-like-the-oracle
  (is (= "\"a\\\\b\\\"c\\n\\r\\t\\u0001é\"" (prep/edn-str "a\\b\"c\n\r\t\u0001é"))))

(deftest config-values-print-like-python
  (testing "str(v) for the JSON scalars: ints bare, floats as repr, strings quoted"
    (is (= "1024" (prep/py-str 1024)))
    (is (= "160000.0" (prep/py-str 160000.0)))
    (is (= "1e-05" (prep/py-str 1.0E-5)))
    (is (= "[\"a\" 1.5 2]" (prep/config-value ["a" 1.5 2])))
    (is (= "{\"choice:2\" 1.9063563346862793}" (prep/config-value {"choice:2" 1.9063563346862793})))))

(deftest missing-checkpoint-says-where-to-get-it
  ;; the usual mistake: ../laya is a checkout of the GitHub laya repo (the
  ;; Python package), not the weights from the Hub
  (let [dir "target/prepare-test-empty"]
    (jolt.host/delete-tree! dir)
    (io/make-parents (io/file dir "x"))
    (spit (str dir "/README.md") "not a checkpoint")
    (let [e (try (prep/convert dir "target/prepare-test-empty-out") nil
                 (catch Exception e e))]
      (is (some? e) "convert must refuse a directory without the checkpoint files")
      (is (= :checkpoint-missing (:type (ex-data e))))
      (is (= ["model.safetensors" "tokenizer/tokenizer.json" "encoder/config.json" "rl_agent_config.json"]
             (:missing (ex-data e))))
      (is (str/includes? (ex-message e) "https://huggingface.co/convaiinnovations/laya"))
      (is (str/includes? (ex-message e) dir)))
    (jolt.host/delete-tree! dir)
    (jolt.host/delete-tree! "target/prepare-test-empty-out")))
