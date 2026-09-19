(ns lev.prepare
  "jolt prepare: convert an encoder checkpoint into data/ (the port of the
  reference Python converter; golden/prepare.edn pins its output and
  test/lev/prepare_test.clj checks this one reproduces it).

  - model.safetensors -> <out>/model/<tensor>.f32, raw little-endian f32.
    F16 tensors are widened by the C kernel (exact); F32 ones copied.
  - <out>/manifest.edn   tensor name -> {:shape :file}
  - tokenizer.json (+ tokenizer_config.json) -> <out>/tokenizer.edn
    {vocab merges specials added}, plus the sentencepiece fields for mmBERT
  - encoder/config.json + rl_agent_config.json -> <out>/config.edn

  Usage: jolt prepare  (task)  or  jolt -M:prepare [--checkpoints DIR] [--out DIR]"
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [jolt.ffi :as ffi]
            [lev.config :as cfg]
            [lev.sequence :as seq])
  (:gen-class))

(ffi/defcfn f16-file->f32* "lev_f16_file_to_f32" [:string :int64 :int64 :string] :int64)
(ffi/defcfn copy-file-range* "lev_copy_file_range" [:string :int64 :int64 :string] :int64)
(ffi/defcfn read-file-range* "lev_read_file_range" [:string :int64 :int64 :pointer] :int64)
(ffi/defcfn file-crc32* "lev_file_crc32" [:string :pointer] :int64)

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

(defn tokenizer-specials
  "{\"cls\" id ...} from tokenizer_config.json's *_token names (plain strings
  or AddedToken objects) looked up among the added tokens."
  [tcfg added]
  (into (sorted-map)
        (for [k ["cls" "sep" "mask" "pad" "unk"]]
          (let [v (get tcfg (str k "_token"))
                name (if (map? v) (get v "content") v)
                id (get added name)]
            (when-not id
              (throw (ex-info (str k "_token " (pr-str name) " is not an added token of this tokenizer")
                              {:key (str k "_token") :name name})))
            [k id]))))

(defn tokenizer-kind
  "What tokenizer.json describes: :byte-level (GPT-2 style: NFC + ByteLevel
  pre-tokenizer) or :sentencepiece (Metaspace + byte fallback, the
  mmBERT/Gemma tokenizer)."
  [tok]
  (case (get-in tok ["pre_tokenizer" "type"])
    "ByteLevel" :byte-level
    "Metaspace" :sentencepiece
    (throw (ex-info (str "unsupported pre_tokenizer " (pr-str (get-in tok ["pre_tokenizer" "type"])))
                    {:pre-tokenizer (get tok "pre_tokenizer")}))))

(defn write-tokenizer
  "tokenizer.json (+ tokenizer_config.json for the special names) ->
  tokenizer.edn. The byte-level format is byte-identical to the reference
  converter's (golden/prepare.edn pins it); the sentencepiece one adds
  :kind, :mask-token, :byte-fallback, :fuse-unk and :lstrip."
  [tok-json tok-cfg-json out]
  (let [tok (json/read-str (slurp tok-json))
        tcfg (json/read-str (slurp tok-cfg-json))
        kind (tokenizer-kind tok)
        vocab (get-in tok ["model" "vocab"])
        merges (get-in tok ["model" "merges"])
        added-tokens (get tok "added_tokens")
        added (into {} (map (fn [t] [(get t "content") (get t "id")])) added-tokens)
        specials (tokenizer-specials tcfg added)
        sb (StringBuilder.)]
    (.append sb (if (= kind :sentencepiece)
                  (str "{:format 2\n :kind :sentencepiece\n"
                       " :mask-token " (edn-str (let [m (get tcfg "mask_token")] (if (map? m) (get m "content") m))) "\n"
                       " :byte-fallback " (boolean (get-in tok ["model" "byte_fallback"])) "\n"
                       " :fuse-unk " (boolean (get-in tok ["model" "fuse_unk"])) "\n"
                       " :lstrip [" (str/join " " (for [t added-tokens :when (get t "lstrip")] (edn-str (get t "content")))) "]\n"
                       " :vocab {\n")
                  "{:format 1\n :vocab {\n"))
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
    (println (format "tokenizer: %s, %d vocab, %d merges, %d added, specials %s"
                     (name kind) (count vocab) (count merges) (count added-tokens) (pr-str specials)))))

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

(def hub-url "https://huggingface.co/convaiinnovations/laya")

(def checkpoint-files
  "What a checkpoint directory must contain (the Hub repo's layout)."
  ["model.safetensors" "tokenizer/tokenizer.json" "tokenizer/tokenizer_config.json"
   "encoder/config.json" "rl_agent_config.json"])

(defn check-checkpoint!
  "Throw {:type :checkpoint-missing} naming the absent files and where to get
  them. The usual mistake is pointing at a checkout of the GitHub repo of the
  Python package,
  which is the Python package, not the weights."
  [home]
  (let [missing (vec (remove #(.exists (io/file home %)) checkpoint-files))]
    (when (seq missing)
      (throw (ex-info (str "no checkpoint under " home " (missing "
                           (str/join ", " missing) ").\n"
                           "The weights live on the Hub (convaiinnovations/laya), not in the package's GitHub repo: download "
                           hub-url " into that directory (README, \"Getting the checkpoint\"), "
                           "or point LEV_CHECKPOINTS / --checkpoints at where you put it.")
                      {:type :checkpoint-missing :checkpoints-home home :missing missing})))))

(def checkpoints
  "Checkpoint name -> its subfolder in the Hub bundle (english is the root).
  Prepared data mirrors the layout: <out>/, <out>/multilingual, <out>/typed-decisions."
  (seq/ordered-map [["english" ""] ["multilingual" "multilingual"] ["typed-decisions" "typed-decisions"]]))


(defn checkpoint-dir [home name]
  (let [sub (get checkpoints name)]
    (if (= sub "") home (str home "/" sub))))

(defn out-dir [out name]
  (let [sub (get checkpoints name)]
    (if (= sub "") out (str out "/" sub))))

(defn- present? [dir]
  (every? #(.exists (io/file dir %)) checkpoint-files))

(defn plan
  "What to convert: [{:name :src :out} ...]. With no names, the root
  (english, required) plus every subfolder that is there; with names, exactly
  those, each of which must be there."
  [home out names]
  (doseq [n names]
    (when-not (contains? checkpoints n)
      (throw (ex-info (str "unknown checkpoint " (pr-str n) "; choose one of " (str/join ", " (keys checkpoints)))
                      {:type :unknown-model :model n :known (vec (keys checkpoints))}))))
  (if (seq names)
    (vec (for [n names]
           (let [src (checkpoint-dir home n)]
             (check-checkpoint! src)
             {:name n :src src :out (out-dir out n)})))
    (do (check-checkpoint! home)
        (vec (for [[n _] checkpoints
                   :let [src (checkpoint-dir home n)]
                   :when (present? src)]
               {:name n :src src :out (out-dir out n)})))))

(defn convert
  "Run the whole conversion of the checkpoint under home into out."
  [home out]
  (check-checkpoint! home)
  (let [entries (convert-weights (str home "/model.safetensors") out)]
    (write-manifest entries out)
    (write-tokenizer (str home "/tokenizer/tokenizer.json") (str home "/tokenizer/tokenizer_config.json") out)
    (write-config (str home "/encoder/config.json") (str home "/rl_agent_config.json") out)
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

(defn convert-bundle
  "Convert the checkpoints `plan` lists. Answers {name entries}."
  [home out names]
  (into {}
        (for [{:keys [name src out]} (plan home out names)]
          (do (println (str "== " name ": " src " -> " out))
              [name (convert src out)]))))

(defn -main
  "jolt -M:prepare [--checkpoints DIR] [--out DIR] [--model NAME]. Falls back to
  LEV_CHECKPOINTS / LEV_DATA, then config.edn :checkpoints-home / :data, then ../laya and
  data. Without --model every checkpoint present under DIR is converted
  (english at the root, multilingual/ and typed-decisions/ when downloaded)."
  [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        home (cfg/setting ctx "--checkpoints" "LEV_CHECKPOINTS" :checkpoints-home "../laya")
        out (cfg/setting ctx "--out" "LEV_DATA" :data "data")
        model (get opts "--model")
        names (when (and (string? model) (not= model "all")) [model])]
    (try (convert-bundle home out names)
         (catch Exception e
           (if (#{:checkpoint-missing :unknown-model} (:type (ex-data e)))
             (do (binding [*out* *err*] (println "jolt prepare:" (ex-message e)))
                 (System/exit 1))
             (throw e))))))
