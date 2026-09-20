(ns lev.mlx
  "The encoder forward on Apple's GPU: MLX through native/lev_mlx.c (the
  `lev_mlx` native, built by `jolt mlx` from mlx-c at a pinned tag;
  optional, mac only, so a tree without it runs the encoders on the C
  kernels as before).

  An MLX agent is an ordinary encoder agent (lev.agent/agent-shell: the
  same config, tokenizer, prefix cache, sequence building, calibration,
  constraints, debias) whose :w is a model handle on the device and whose
  :forward is `forward-batch` here, the same contract as
  lev.model/forward-batch: prepared rows in, [[logits act-logits] ...]
  out, f32. The weights come from the same prepared data/ directory, held
  on the device at :dtype :f32 (the default: the goldens' precision, what
  lev.mlx-test holds it to) or :f16 (half the memory, faster; the argmax
  agrees, the values drift by up to 1e-2, so not for the golden path).

  laya-mlx (the independent MLX port of the same checkpoints) measured
  the same architecture at 13 ms a short question at f16 on an M3 Max
  against ~120 ms on lev's C kernels; bench/paired.clj --candidates
  cpu,mlx measures it here."
  (:require [jolt.ffi :as ffi]
            [lev.agent :as ag]))

;; :blocking (the collector may run while a call is inside C) rules out
;; :string arguments: every string goes in as an arena-owned pointer
(ffi/defcfn new* "lev_mlx_new"
  [:int :int :int :int :int :int :int :int :float :float :float :pointer :int] :pointer)
(ffi/defcfn ok* "lev_mlx_ok" [:pointer] :int)
(ffi/defcfn error* "lev_mlx_error" [:pointer] :string)
(ffi/defcfn gpu* "lev_mlx_gpu" [:pointer] :int)
(ffi/defcfn version* "lev_mlx_version" [] :string)
(ffi/defcfn load-tensor* "lev_mlx_load_tensor" [:pointer :pointer :pointer :int64 :int64] :int :blocking)
(ffi/defcfn complete* "lev_mlx_complete" [:pointer :pointer :int] :int)
(ffi/defcfn forward* "lev_mlx_forward"
  [:pointer :pointer :pointer :int :int :pointer :pointer :int :pointer :pointer :pointer] :int :blocking)
(ffi/defcfn set-selected-head* "lev_mlx_set_selected_head" [:pointer :int] :void)
(ffi/defcfn free* "lev_mlx_free" [:pointer] :void)
(ffi/defcfn memory* "lev_mlx_memory" [:int] :int64)

(defn available?
  "Is the mlx native loaded? (jolt mlx builds it; mac only.)"
  []
  (some? (ffi/find-symbol "lev_mlx_forward")))

(defn version [] (version*))

(defn- unavailable [dir]
  (throw (ex-info "the mlx native is not built: run jolt mlx (mac only; see README, Native dependencies)"
                  {:type :model-unavailable :model dir :backend :mlx})))

(defn- fail [h dir msg]
  (when-not (zero? (ffi/address h)) (free* h))
  (throw (ex-info (str "cannot load " dir " on mlx: " msg
                       (when (re-find #"metallib" (str msg))
                         " (MLX loads its Metal kernels from mlx.metallib next to the binary holding it: native/mlx.metallib for jolt run/test, which jolt mlx writes; copy it next to lev-server for a jolt build)"))
                  {:type :model-unavailable :model dir :backend :mlx})))

(defn load-model
  "The model in a prepared data directory, on the device: {:p handle :gpu?
  :dtype :selected-head}. cfg and manifest as lev.agent/agent-shell reads
  them; selected-head? prunes the last head layer to the CLS + marker rows
  (exact; lev.model's default too). Throws {:type :model-unavailable}
  when the native is missing or a tensor is."
  [data-dir cfg manifest dtype selected-head?]
  (when-not (available?) (unavailable data-dir))
  (let [{:keys [hidden-size num-layers num-heads head-dim intermediate window head-layers vocab-size
                rope-full rope-local norm-eps layer-types]} cfg
        h (ffi/with-arena [a]
            (let [sliding (ffi/alloc a (* 4 (count layer-types)))]
              (doseq [[i t] (map-indexed vector layer-types)]
                (ffi/write sliding :int32 (int (if (= t "sliding_attention") 1 0)) (* 4 i)))
              (new* (int hidden-size) (int num-layers) (int num-heads) (int head-dim) (int intermediate)
                    (int window) (int head-layers) (int vocab-size)
                    (float rope-full) (float rope-local) (float norm-eps)
                    sliding (int (if (= dtype :f16) 1 0)))))]
    (when (zero? (ffi/address h)) (fail h data-dir "out of memory"))
    (when (zero? (ok* h)) (fail h data-dir (error* h)))
    (doseq [[name {:keys [shape file]}] (:tensors manifest)]
      (let [[rows cols] shape
            rc (ffi/with-arena [a]
                 (load-tensor* h (ffi/string->ptr a (str name)) (ffi/string->ptr a (str data-dir "/" file))
                               (long rows) (long (or cols 0))))]
        (when (neg? rc) (fail h data-dir (error* h)))))
    (let [missing (ffi/with-arena [a]
                    (let [buf (ffi/alloc a 256)]
                      (when (zero? (complete* h buf 256)) (ffi/ptr->string buf))))]
      (when missing (fail h data-dir (str "the manifest has no " missing))))
    ;; the last head layer's out-projection and FFN on the CLS + marker
    ;; rows only, as lev.model's forward-batch does
    (set-selected-head* h (int (if selected-head? 1 0)))
    {:p h :gpu? (= 1 (gpu* h)) :dtype dtype :selected-head selected-head?}))

(defn free!
  "Give the model back to the device."
  [model]
  (free* (:p model))
  nil)

(defn memory
  "Bytes MLX holds for arrays; with clear?, its cache of freed buffers is
  released first (after unloading a model)."
  ([] (memory false))
  ([clear?] (memory* (int (if clear? 1 0)))))

(defn forward-batch
  "B rows through the model, as lev.model/forward-batch: each row {:ids
  [L_b] :att [L_b] :markers [k_b] :marker-mask [k_b] :qtype q}, padded to
  the longest here; answers [[logits act-logits] ...] as float vectors,
  one logit per marker (-1e4 where its mask is 0)."
  [model cfg rows]
  (let [B (count rows)
        L (reduce max (map #(count (:ids %)) rows))
        kmax (max 2 (reduce max (map #(count (:markers %)) rows)))
        pad (or (:pad-id cfg) 0)]
    (ffi/with-arena [a]
      (let [ids (ffi/alloc a (* 4 B L))
            att (ffi/alloc a (* B L))
            mpos (ffi/alloc a (* 4 B kmax))
            mmask (ffi/alloc a (* B kmax))
            qtype (ffi/alloc a (* 4 B))
            logits (ffi/alloc a (* 4 B kmax))
            act (ffi/alloc a (* 4 B 2))]
        (doseq [[b row] (map-indexed vector rows)]
          (let [n (count (:ids row))]
            (dotimes [i L]
              (ffi/write ids :int32 (int (if (< i n) (nth (:ids row) i) pad)) (* 4 (+ (* b L) i)))
              (ffi/write att :uint8 (byte (if (< i n) (nth (:att row) i) 0)) (+ (* b L) i)))
            (dotimes [k kmax]
              (let [have (< k (count (:markers row)))]
                (ffi/write mpos :int32 (int (if have (max 0 (long (nth (:markers row) k))) 0)) (* 4 (+ (* b kmax) k)))
                (ffi/write mmask :uint8 (byte (if have (nth (:marker-mask row) k) 0)) (+ (* b kmax) k))))
            (ffi/write qtype :int32 (int (:qtype row)) (* 4 b))))
        (when-not (zero? (forward* (:p model) ids att (int B) (int L) mpos mmask (int kmax) qtype logits act))
          (throw (ex-info (str "mlx forward failed: " (error* (:p model)))
                          {:type :forward-failed :backend :mlx :rows B :length L})))
        (mapv (fn [b row]
                (let [k (count (:markers row))]
                  [(mapv (fn [i] (ffi/read logits :float (* 4 (+ (* b kmax) i)))) (range k))
                   [(ffi/read act :float (* 4 (* b 2))) (ffi/read act :float (* 4 (inc (* b 2))))]]))
              (range B) rows)))))

(defn load-agent
  "An encoder agent on MLX for a prepared data directory: lev.agent's
  shell (limits, name) with the weights on the device at :dtype (:f32
  default, :f16) and this namespace's forward; :selected-head false in
  the limits (or the prepared config) keeps the last head layer whole.
  :close (the router calls it on eviction) frees the model. Throws
  {:type :model-unavailable}."
  ([data-dir] (load-agent data-dir nil))
  ([data-dir limits]
   (when-not (available?) (unavailable data-dir))
   (let [dtype (or (:dtype limits) :f32)
         shell (ag/agent-shell data-dir limits)
         selected? (not= false (:selected-head limits (:selected-head (:cfg shell))))
         model (load-model data-dir (:cfg shell) (:manifest shell) dtype selected?)
         freed (atom false)]
     (assoc shell
            :backend :mlx :dtype dtype :gpu? (:gpu? model)
            :w model
            :forward forward-batch
            :close (fn [a] (when (compare-and-set! freed false true) (free! (:w a))))))))
