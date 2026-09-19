(ns laya.prepare
  "jolt prepare: convert the Laya checkpoint into data/ (the port of the
  reference Python converter; golden/prepare.edn pins its output and
  test/laya/prepare_test.clj checks this one reproduces it).

  - model.safetensors -> <out>/model/<tensor>.f32, raw little-endian f32.
    F16 tensors are widened by the C kernel (exact); F32 ones copied.
  - <out>/manifest.edn   tensor name -> {:shape :file}
  - tokenizer.json       -> <out>/tokenizer.edn {vocab merges specials added}
  - encoder/config.json + rl_agent_config.json -> <out>/config.edn

  Usage: jolt prepare  (task)  or  jolt -M:prepare [--laya DIR] [--out DIR]"
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [jolt.ffi :as ffi]
            [laya.sequence :as seq])
  (:gen-class))

(ffi/defcfn f16-file->f32* "lla_f16_file_to_f32" [:string :int64 :int64 :string] :int64)
(ffi/defcfn copy-file-range* "lla_copy_file_range" [:string :int64 :int64 :string] :int64)
(ffi/defcfn read-file-range* "lla_read_file_range" [:string :int64 :int64 :pointer] :int64)
(ffi/defcfn file-crc32* "lla_file_crc32" [:string :pointer] :int64)

;; --- edn writing, byte-compatible with the python oracle ---------------------

(defn edn-str
  "A quoted EDN string: escape \\\\ \" \\n \\r \\t, other controls as \\uXXXX,
  everything else verbatim (UTF-8)."
  [s]
  (let [sb (StringBuilder. "\"")]
    (doseq [c s]
      (cond
        (= c \\) (.append sb "\\\\")
        (= c \") (.append sb "\\\"")
        (= c \newline) (.append sb "\\n")
        (= c \return) (.append sb "\\r")
        (= c \tab) (.append sb "\\t")
        (< (int c) 32) (.append sb (format "\\u%04x" (int c)))
        :else (.append sb c)))
    (.append sb "\"")
    (.toString sb)))

(defn py-str
  "Python str() of a JSON scalar: ints bare, floats as float repr."
  [v]
  (cond
    (string? v) v
    (integer? v) (str v)
    (number? v) (seq/py-float-str v)
    (boolean? v) (if v "True" "False")
    (nil? v) "None"
    :else (str v)))

(defn config-value
  "config.edn value: strings quoted, vectors of quoted-strings-or-scalars,
  maps of quoted key -> scalar, scalars via str()."
  [v]
  (cond
    (string? v) (edn-str v)
    (sequential? v) (str "[" (str/join " " (map #(if (string? %) (edn-str %) (py-str %)) v)) "]")
    (map? v) (str "{" (str/join " " (map (fn [[k x]] (str (edn-str k) " " (py-str x))) v)) "}")
    :else (py-str v)))

;; --- safetensors ----------------------------------------------------------------

(defn read-header
  "[header-map data-base-offset] of a safetensors file: 8-byte LE length,
  then that many bytes of JSON."
  [path]
  (with-open [a (ffi/confined-arena)]
    (let [p8 (ffi/alloc a 8)]
      (when (not= 8 (read-file-range* path 0 8 p8))
        (throw (ex-info "cannot read safetensors header" {:path path})))
      (let [n (ffi/read p8 :int64 0)
            pj (ffi/alloc a n)]
        (when (not= n (read-file-range* path 8 n pj))
          (throw (ex-info "truncated safetensors header" {:path path :n n})))
        [(json/read-str (ffi/read-bytes pj n)) (+ 8 n)]))))

(defn- checked [ret what]
  (when (neg? ret) (throw (ex-info (str "prepare: " what " failed") {})))
  ret)

(defn convert-weights
  "Every tensor -> <out>/model/<name>.f32. Answers [[name shape file] ...]
  sorted by name (the manifest order)."
  [st-path out]
  (let [[hdr base] (read-header st-path)
        entries (sort-by first compare (dissoc hdr "__metadata__"))]
    (io/make-parents (io/file out "model" "x"))
    (vec
     (for [[name meta] entries]
       (let [dtype (get meta "dtype")
             shape (get meta "shape")
             [o0 o1] (get meta "data_offsets")
             n (reduce * 1 shape)
             file (str "model/" name ".f32")
             dst (str out "/" file)]
         (case dtype
           "F16" (checked (f16-file->f32* st-path (+ base o0) n dst) (str "f16->f32 of " name))
           "F32" (checked (copy-file-range* st-path (+ base o0) (- o1 o0) dst) (str "copy of " name))
           (throw (ex-info (str "unexpected dtype " dtype " for " name) {:name name :dtype dtype})))
         (println (format "  %-48s %-16s %s" name (str "[" (str/join ", " shape) "]") dtype))
         [name shape file])))))

(defn write-manifest [entries out]
  (spit (str out "/manifest.edn")
        (str "{:format 1\n :tensors {\n"
             (apply str (for [[name shape file] entries]
                          (str "  " (edn-str name) " {:shape [" (str/join " " shape) "] :file " (edn-str file) "}\n")))
             "}}\n")))

;; --- tokenizer --------------------------------------------------------------------

(defn write-tokenizer [tok-json out]
  (let [tok (json/read-str (slurp tok-json))
        vocab (get-in tok ["model" "vocab"])
        merges (get-in tok ["model" "merges"])
        added-tokens (get tok "added_tokens")
        added (into {} (map (fn [t] [(get t "content") (get t "id")])) added-tokens)
        specials (into (sorted-map)
                       (map (fn [k] [k (get added (str "[" (str/upper-case k) "]"))]))
                       ["cls" "sep" "pad" "mask" "unk"])
        sb (StringBuilder.)]
    (.append sb "{:format 1\n :vocab {\n")
    (doseq [[t i] (sort-by val vocab)]
      (.append sb "  ") (.append sb (edn-str t)) (.append sb " ") (.append sb (str i)) (.append sb "\n"))
    (.append sb " }\n :merges [\n")
    (doseq [m merges]
      (let [[a b] (if (string? m) (str/split m #" ") m)]
        (.append sb "  ") (.append sb (edn-str (str a " " b))) (.append sb "\n")))
    (.append sb " ]\n :specials {\n")
    (doseq [[k i] specials]
      (.append sb "  :") (.append sb k) (.append sb " ") (.append sb (str i)) (.append sb "\n"))
    (.append sb " }\n :added [\n")
    (doseq [t (sort-by #(get % "id") added-tokens)]
      (.append sb "  [") (.append sb (edn-str (get t "content"))) (.append sb " ")
      (.append sb (str (get t "id"))) (.append sb "]\n"))
    (.append sb " ]}\n")
    (spit (str out "/tokenizer.edn") (.toString sb))
    (println (format "tokenizer: %d vocab, %d merges, %d added, specials %s"
                     (count vocab) (count merges) (count added-tokens) (pr-str specials)))))

;; --- config -------------------------------------------------------------------------

(defn write-config [enc-json rl-json out]
  (let [enc (json/read-str (slurp enc-json))
        rl (json/read-str (slurp rl-json))
        hidden (get enc "hidden_size")
        cfg [[":hidden-size" hidden]
             [":num-layers" (get enc "num_hidden_layers")]
             [":num-heads" (get enc "num_attention_heads")]
             [":head-dim" (quot hidden (get enc "num_attention_heads"))]
             [":intermediate" (get enc "intermediate_size")]
             ;; sliding-attention radius = local_attention // 2
             ;; (torch: config.sliding_window = local_attention // 2)
             [":window" (quot (get enc "local_attention") 2)]
             [":layer-types" (get enc "layer_types")]
             [":rope-full" (get-in enc ["rope_parameters" "full_attention" "rope_theta"])]
             [":rope-local" (get-in enc ["rope_parameters" "sliding_attention" "rope_theta"])]
             [":norm-eps" (get enc "norm_eps")]
             [":vocab-size" (get enc "vocab_size")]
             [":max-len" (get rl "max_len")]
             [":head-max-len" (get rl "head_max_len")]
             [":head-layers" (get rl "head_layers")]
             [":head-ffn" (* 4 hidden)]
             [":temperature" (get rl "temperature")]
             [":temperature-by-options" (get rl "temperature_by_options" {})]]]
    (spit (str out "/config.edn")
          (str "{\n" (apply str (for [[k v] cfg] (str " " k " " (config-value v) "\n"))) "}\n"))))

;; --- driver ----------------------------------------------------------------------------

(defn convert
  "Run the whole conversion of the checkpoint under laya-home into out."
  [laya-home out]
  (let [entries (convert-weights (str laya-home "/model.safetensors") out)]
    (write-manifest entries out)
    (write-tokenizer (str laya-home "/tokenizer/tokenizer.json") out)
    (write-config (str laya-home "/encoder/config.json") (str laya-home "/rl_agent_config.json") out)
    (println (format "config.edn written; %d tensors, total %.2f GB"
                     (count entries)
                     (/ (* 4.0 (reduce + (map (fn [[_ shape _]] (reduce * 1 shape)) entries))) 1e9)))
    entries))

(defn file-crc32
  "[size crc32] of a file, zlib-compatible (the prepare parity oracle)."
  [path]
  (with-open [a (ffi/confined-arena)]
    (let [sz (ffi/alloc a 8)
          crc (file-crc32* path sz)]
      (when (neg? crc) (throw (ex-info "cannot read file" {:path path})))
      [(ffi/read sz :int64 0) crc])))

(defn -main [& args]
  (let [opts (apply hash-map args)
        laya-home (or (get opts "--laya") (System/getenv "LAYA_HOME") "../laya")
        out (or (get opts "--out") "data")]
    (convert laya-home out)))
