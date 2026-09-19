(ns lev.llm
  "llama.cpp through native/lev_llm.c (the `lev_llm` native, built by
  `jolt llama`; optional, so a tree without it runs the encoders alone).

  A model is a handle on one GGUF and one context. `generate` completes a
  prompt; `decide` is what the thinker engine runs per question: decode a
  chat prompt, let the model think until its closing tag, force an answer
  prefix, then score each candidate answer by teacher-forcing its tokens
  on a copy of the KV state, answering one log probability per option.
  `chat-prompt` renders ChatML with the model's thinking switch, as
  MiniCPM5's template does (`<think>\\n` to think, `<think>\\n\\n</think>\\n\\n`
  not to)."
  (:require [clojure.string :as str]
            [jolt.ffi :as ffi]))

;; :blocking (the collector may run while a call is inside C, seconds for a
;; thought) rules out :string arguments: every string goes in as an
;; arena-owned pointer that outlives the call
(ffi/defcfn load* "lev_llm_load" [:pointer :int :int :int :int] :pointer :blocking)
(ffi/defcfn free* "lev_llm_free" [:pointer] :void)
(ffi/defcfn ok* "lev_llm_ok" [:pointer] :int)
(ffi/defcfn error* "lev_llm_error" [:pointer] :string)
(ffi/defcfn version* "lev_llm_version" [] :string)
(ffi/defcfn n-ctx* "lev_llm_n_ctx" [:pointer] :int)
(ffi/defcfn n-seq-max* "lev_llm_n_seq_max" [:pointer] :int)
(ffi/defcfn count-tokens* "lev_llm_count_tokens" [:pointer :string] :int)
(ffi/defcfn generate* "lev_llm_generate"
  [:pointer :pointer :int :pointer :float :float :float :uint32 :pointer :int] :int :blocking)
(ffi/defcfn decide* "lev_llm_decide"
  [:pointer :pointer :int :pointer :pointer :pointer :int :pointer :float :float :float :uint32 :pointer :pointer :int]
  :int :blocking)

(defn available?
  "Is the llm native loaded? (jolt llama builds it.)"
  []
  (some? (ffi/find-symbol "lev_llm_load")))

(defn version [] (version*))

(def defaults
  {:n-ctx 4096 :n-gpu-layers -1 :threads 0 :n-seq-max 32
   :temperature 1.0 :top-p 0.95 :min-p 0.0 :seed 42
   :think-max 2048 :think-end "</think>" :answer-prefix "\n\nANSWER: " :answer-end "<|im_end|>"})

(defn load
  "Load a GGUF: {:n-ctx (4096) :n-gpu-layers (-1 = all) :threads (0 =
  llama.cpp's) :n-seq-max (32, the most options one question can have)}.
  Throws {:type :model-unavailable} when the native is missing or the
  file cannot be loaded."
  [path opts]
  (when-not (available?)
    (throw (ex-info "the llm native is not built: run jolt llama (see README, Native dependencies)"
                    {:type :model-unavailable :model path})))
  (let [{:keys [n-ctx n-gpu-layers threads n-seq-max]} (merge defaults opts)
        h (ffi/with-arena [a]
            (load* (ffi/string->ptr a (str path)) (int n-ctx) (int n-gpu-layers) (int threads) (int n-seq-max)))]
    (when (or (zero? (ffi/address h)) (zero? (ok* h)))
      (let [msg (if (zero? (ffi/address h)) "out of memory" (error* h))]
        (when-not (zero? (ffi/address h)) (free* h))
        (throw (ex-info (str "cannot load " path ": " msg) {:type :model-unavailable :model path}))))
    {:p h :path (str path) :opts (merge defaults opts)}))

(defn ok? [m] (and m (pos? (ok* (:p m)))))
(defn n-ctx [m] (n-ctx* (:p m)))
(defn n-seq-max [m] (n-seq-max* (:p m)))
(defn free! [m] (free* (:p m)) nil)
(defn count-tokens [m text] (count-tokens* (:p m) text))

(defn chat-prompt
  "ChatML for `messages` ({:role :content}, string keys accepted) through
  the assistant turn. opts :thinking (true: the open thought tag, for
  `decide` to close; false: the closed empty thought, so the model
  answers at once; nil: neither). Either way what follows the closing
  tag is the answer prefix (\"\\n\\nANSWER: \" by default), so a decided
  answer is always scored after `</think>\\n\\nANSWER: `, the shape the
  template produces."
  [_m messages {:keys [thinking]}]
  (let [turn (fn [{:keys [role content] :as msg}]
               (str "<|im_start|>" (or role (get msg "role")) "\n" (or content (get msg "content")) "<|im_end|>\n"))]
    (str (apply str (map turn messages))
         "<|im_start|>assistant\n"
         (case thinking
           true "<think>\n"
           false "<think>\n\n</think>"
           ""))))

(defn generate
  "Complete `prompt`: {:max-tokens (256) :stop (nil) :temperature (model
  default; 0 = greedy) :top-p :min-p :seed} -> {:text :tokens}."
  [m prompt {:keys [max-tokens stop temperature top-p min-p seed]
             :or {max-tokens 256}}]
  (let [{:keys [opts]} m
        cap (* 8 (+ max-tokens 16))]
    (ffi/with-arena [a]
      (let [out (ffi/alloc a cap)
            n (generate* (:p m) (ffi/string->ptr a prompt) (int max-tokens) (ffi/string->ptr a (or stop ""))
                         (float (or temperature (:temperature opts))) (float (or top-p (:top-p opts)))
                         (float (or min-p (:min-p opts))) (long (or seed (:seed opts))) out (int cap))]
        (when (neg? n)
          (throw (ex-info (str "generation failed: " (error* (:p m))) {:type :llm-error})))
        {:text (ffi/ptr->string out) :tokens n}))))

(defn decide
  "Score `options` as answers to `prompt` (a chat-prompt): {:think-max
  (tokens of thought; 0 = none, the prompt must then close the thought)
  :think-end :answer-prefix :answer-end :temperature :top-p :min-p :seed}
  -> {:logp [per option] :thought text :tokens thought-tokens}."
  [m prompt options {:keys [think-max think-end answer-prefix answer-end temperature top-p min-p seed]}]
  (let [opts (:opts m)
        n (count options)
        think-max (long (or think-max (:think-max opts)))
        think-cap (* 8 (+ think-max 64))]
    (ffi/with-arena [a]
      (let [optv (ffi/alloc a (* 8 n))
            strs (mapv (fn [o] (ffi/string->ptr a (str o))) options)
            _ (dotimes [i n] (ffi/write optv :pointer (nth strs i) (* 8 i)))
            logp (ffi/alloc a (* 8 n))
            thought (ffi/alloc a think-cap)
            made (decide* (:p m) (ffi/string->ptr a prompt) (int think-max)
                          (ffi/string->ptr a (or think-end (:think-end opts)))
                          (ffi/string->ptr a (or answer-prefix (:answer-prefix opts)))
                          optv (int n)
                          (ffi/string->ptr a (or answer-end (:answer-end opts)))
                          (float (or temperature (:temperature opts))) (float (or top-p (:top-p opts)))
                          (float (or min-p (:min-p opts))) (long (or seed (:seed opts)))
                          logp thought (int think-cap))]
        (when (neg? made)
          (throw (ex-info (str "decide failed: " (error* (:p m))) {:type :llm-error})))
        {:logp (mapv #(ffi/read logp :double (* 8 %)) (range n))
         :thought (ffi/ptr->string thought)
         :tokens made}))))

(defn argmax [xs]
  (reduce (fn [bi i] (if (> (nth xs i) (nth xs bi)) i bi)) 0 (range (count xs))))

(defn softmax
  "Probabilities from log scores."
  [logp]
  (let [mx (reduce max logp)
        ex (mapv #(Math/exp (- (double %) mx)) logp)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))
