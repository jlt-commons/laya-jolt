(ns laya.tensors
  "f32 tensors as (ffi) buffer views + the C kernel/BLAS ops the model needs.

  A tensor is {:p ptr :shape [rows cols] :size n} — raw pointers into ffi
  buffers, row-major, little-endian f32. Matmuls call cblas_sgemm through
  the blas native (Accelerate on mac, OpenBLAS on linux); the elementwise
  and reduction ops are native/laya_kernels.c. Nothing per-element crosses
  back into Clojure."
  (:require [clojure.java.io :as io]
            [jolt.ffi :as ffi]))

;; --- cblas (Accelerate / OpenBLAS) -------------------------------------------
;; CBLAS_ROW_MAJOR=101. sgemm computes C = alpha*A*B + beta*C, all row-major
;; f32. Our matmuls are always out = X @ W^T with W stored [out-dim x in-dim]
;; (torch Linear convention), so W is transposed via the TransB=1 flag.

(ffi/defcfn cblas-sgemm* "cblas_sgemm"
  [:int :int :int :int64 :int64 :int64
   :float :pointer :int64 :pointer :int64
   :float :pointer :int64] :void)

(def ^:private RowMajor 101)
(def ^:private NoTrans 111)
(def ^:private Trans 112)

;; --- kernels (native/laya_kernels.c) ------------------------------------------

(ffi/defcfn gather-rows* "lla_gather_rows"
  [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn layernorm* "lla_layernorm"
  [:pointer :pointer :pointer :int64 :int64 :float :pointer] :void)
(ffi/defcfn gelu* "lla_gelu" [:pointer :int64 :pointer] :void)
(ffi/defcfn relu* "lla_relu" [:pointer :int64 :pointer] :void)
(ffi/defcfn silu* "lla_silu" [:pointer :int64 :pointer] :void)
(ffi/defcfn swiglu* "lla_swiglu" [:pointer :int64 :int64 :pointer] :void)
(ffi/defcfn rope-tables* "lla_rope_tables"
  [:double :int64 :int64 :pointer :pointer] :void)
(ffi/defcfn rope-apply* "lla_rope_apply"
  [:pointer :pointer :pointer :int64 :int64 :int64] :void)
(ffi/defcfn split-heads* "lla_split_heads"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn merge-heads* "lla_merge_heads"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn allowed-mask* "lla_allowed_mask"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn masked-softmax* "lla_masked_softmax"
  [:pointer :pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn add-scaled* "lla_add_scaled"
  [:pointer :pointer :int64 :int64 :float :pointer] :void)
(ffi/defcfn add-qtype-bias* "lla_add_qtype_bias"
  [:pointer :pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn softmax* "lla_softmax" [:pointer :int64 :int64 :pointer] :void)

;; --- tensor plumbing ----------------------------------------------------------

(defn ptr [t] (:p t))
(defn shape [t] (:shape t))
(defn size [t] (:size t))

(def ^:dynamic *arena*
  "When bound to an ffi arena, every tensor made here is owned by it and is
  released when the arena closes. A forward pass allocates hundreds of MB of
  intermediates; model/forward-row binds one arena per row so none of it
  outlives the call. Unbound (nil) means caller-owned malloc, which is what
  weights and cached rope tables want."
  nil)

(defn alloc-bytes
  "Zeroed native memory, owned by *arena* when one is bound."
  [n]
  (if *arena* (ffi/alloc *arena* n) (ffi/alloc n)))

(defn make
  [shape]
  {:p (alloc-bytes (* 4 (apply * shape)))
   :shape (vec shape)
   :size (apply * shape)})

(defn get*
  "Read element i (pointer, not tensor)."
  [p i]
  (ffi/read p :float (* 4 i)))

(defn set*
  [p i v]
  (ffi/write p :float (float v) (* 4 i)))

(defn from-floats
  [xs]
  (let [n (count xs)
        t (make [n])]
    (dotimes [i n]
      (set* (:p t) i (nth xs i)))
    t))

(defn from-ints
  "int64 buffer for ids (embedding lookup indices)."
  [xs]
  (let [n (count xs)
        t {:p (alloc-bytes (* 8 n)) :shape [n] :size n}]
    (dotimes [i n]
      (ffi/write (:p t) :int64 (long (nth xs i)) (* 8 i)))
    t))

(defn from-bytes
  "uint8 buffer from 0/1 values."
  ([xs] (from-bytes xs [(count xs)]))
  ([xs shape]
   (let [n (count xs)
         t {:p (alloc-bytes n) :shape (vec shape) :size (apply * shape)}]
     (dotimes [i n]
       (ffi/write (:p t) :uint8 (byte (nth xs i)) i))
     t)))

(defn to-floats
  [t]
  (mapv #(get* (:p t) %) (range (:size t))))

(defn load-file
  "Read a raw f32 file into a fresh tensor of the given shape."
  [path shape]
  (let [n (apply * shape)
        f (io/file path)
        expected (* 4 n)
        len (.length f)]
    (when-not (= len expected)
      (throw (ex-info "file size mismatch" {:path path :expected expected :got len})))
    ;; io/file + Files/readAllBytes shim: copy the raw bytes then write-array.
    ;; Weights live for the whole process: never arena-owned.
    (let [t (binding [*arena* nil] (make shape))
          bytes (java.nio.file.Files/readAllBytes (.toPath (io/file path)))]
      (ffi/write-array (:p t) bytes)
      t)))

(defn load-tensor
  "Load one named tensor from data/manifest.edn."
  [manifest data-dir name]
  (let [{:keys [shape file]} (get-in manifest [:tensors name])]
    (when-not shape
      (throw (ex-info "tensor not in manifest" {:name name})))
    (load-file (str data-dir "/" file) shape)))

;; --- ops ----------------------------------------------------------------------

(defn mmul
  "out = X @ W^T. X [m x k], W [n x k] -> out [m x n]. torch Linear."
  [X W]
  (let [[m k] (:shape X)
        [n k2] (:shape W)]
    (when (or (nil? n) (not= k k2))
      (throw (ex-info "mmul shape mismatch" {:x (:shape X) :w (:shape W)})))
    (let [out (make [m n])]
      (cblas-sgemm* RowMajor NoTrans Trans
                    (long m) (long n) (long k)
                    1.0 (:p X) (long k)
                    (:p W) (long k)
                    0.0 (:p out) (long n))
      out)))

(defn embeddings
  "Gather rows then LayerNorm (weight-only, norm_bias=false) exactly as
  ModernBertEmbeddings: norm(tok_embeddings(ids))."
  [emb-w ids n d norm-w]
  (let [g (make [n d])
        out (make [n d])]
    (gather-rows* (:p emb-w) (:p ids) (long n) (long d) (:p g))
    (layernorm* (:p g) (:p norm-w) 0 (long n) (long d) 1e-5 (:p out))
    out))

(defn gather
  "Row gather: out[i,:] = src[ids[i],:]."
  [src ids n d]
  (let [out (make [n d])]
    (gather-rows* (:p src) (:p ids) (long n) (long d) (:p out))
    out))

(defn layernorm
  "LayerNorm with optional bias (torch nn.LayerNorm)."
  ([x w eps] (layernorm x w 0 eps))
  ([x w b eps]
   (let [[n d] (:shape x)
         out (make [n d])
         bp (if (map? b) (long (:p b)) 0)]
     (layernorm* (:p x) (:p w) bp (long n) (long d) (float eps) (:p out))
     out)))

(defn gelu [x]
  (let [n (:size x) out (make (:shape x))]
    (gelu* (:p x) (long n) (:p out))
    out))

(defn relu [x]
  (let [n (:size x) out (make (:shape x))]
    (relu* (:p x) (long n) (:p out))
    out))

(defn swiglu
  "in [n x 2*mid] -> out [n x mid], act(input)*gate with erf-gelu."
  [x mid]
  (let [[n _] (:shape x) out (make [n mid])]
    (swiglu* (:p x) (long n) (long mid) (:p out))
    out))

(defn rope-tables
  [theta d len]
  (let [cos-t (make [len d])
        sin-t (make [len d])]
    (rope-tables* (double theta) (long d) (long len) (:p cos-t) (:p sin-t))
    [cos-t sin-t]))

(defn rope-apply!
  "Apply rope to head-major q (and k): modifies q in place."
  [q cos-t sin-t n-heads L d]
  (rope-apply* (:p q) (:p cos-t) (:p sin-t) (long n-heads) (long L) (long d))
  q)

(defn split-heads
  [x L H hd]
  (let [out (make [(* H L) hd])]
    (split-heads* (:p x) (long L) (long H) (long hd) (:p out))
    out))

(defn merge-heads
  [src H L hd]
  (let [out (make [L (* H hd)])]
    (merge-heads* (:p src) (long H) (long L) (long hd) (:p out))
    out))

(defn byte-matrix
  "uint8 matrix [r x c] in a raw byte buffer."
  [r c]
  {:p (alloc-bytes (* r c)) :shape [r c] :size (* r c)})

(defn allowed-mask
  "Allowed matrix [L x L] for batch row b: full when window<0, else |i-j|<=window."
  [att b L window]
  (let [m (byte-matrix L L)]
    (allowed-mask* (:p att) (long b) (long L) (long window) (:p m))
    m))

(defn- at-offset
  "Pointer `bytes` past the start of tensor t (a view, not a copy)."
  [t bytes]
  (ffi/segment (+ (ffi/address (:p t)) bytes)))

(defn masked-softmax
  "Row softmax over the allowed keys of a [rows x cols] score block; the
  mask is the top-left [rows x cols] of `allowed` (leading dimension cols
  here; `attention` passes a view into an [L x L] mask)."
  [scores allowed rows cols]
  (let [out (make [rows cols])]
    (masked-softmax* (:p scores) (:p allowed) (long cols) (long rows) (long cols) (:p out))
    out))

(defn attention
  "softmax(scale * Q K^T over the allowed keys) V for every head. q/k/v
  head-major [H*L x hd] (lla_split_qkv's layout), allowed [L x L] bytes ->
  token-major ctx [L x (H*hd)].

  The two products per head are sgemm calls, the same two gemms torch's
  math-path SDPA runs: each head's q/k/v block is already contiguous, and
  P V lands straight in the head's columns of ctx through ldc = H*hd. Only
  the masked softmax is a C loop. A query row with no allowed key (padding
  under a window) stays zero, as lla_masked_softmax leaves its P row zero.

  With a `window` (a sliding layer) the queries go in blocks of 2*window
  rows and each block only scores the keys within window of it, so an
  L=1024 layer touches 256 keys per query instead of 1024. The mask is
  still applied inside the block, so the band is only a saving, never the
  semantics; the full path is one block of everything."
  ([qh kh vh allowed H L hd scale] (attention qh kh vh allowed H L hd scale -1))
  ([qh kh vh allowed H L hd scale window]
   (let [d (* H hd)
         ctx (make [L d])
         block (if (neg? window) L (max 1 (* 2 window)))
         reach (if (neg? window) 0 window)
         max-cols (min L (+ block (* 2 reach)))
         S (make [(min L block) max-cols])
         P (make [(min L block) max-cols])
         head-bytes (* 4 L hd)]
     (dotimes [h H]
       (loop [i0 0]
         (when (< i0 L)
           (let [i1 (min L (+ i0 block))
                 k0 (max 0 (- i0 reach))
                 k1 (min L (+ i1 reach))
                 rows (- i1 i0)
                 cols (- k1 k0)
                 qoff (+ (* h head-bytes) (* 4 i0 hd))
                 koff (+ (* h head-bytes) (* 4 k0 hd))]
             (cblas-sgemm* RowMajor NoTrans Trans
                           (long rows) (long cols) (long hd)
                           (float scale) (at-offset qh qoff) (long hd)
                           (at-offset kh koff) (long hd)
                           0.0 (:p S) (long cols))
             (masked-softmax* (:p S) (at-offset allowed (+ (* i0 L) k0)) (long L)
                              (long rows) (long cols) (:p P))
             (cblas-sgemm* RowMajor NoTrans NoTrans
                           (long rows) (long hd) (long cols)
                           1.0 (:p P) (long cols)
                           (at-offset vh koff) (long hd)
                           0.0 (at-offset ctx (* 4 (+ (* i0 d) (* h hd)))) (long d))
             (recur i1)))))
     ctx)))

(defn add-qtype!
  "h += type_emb[qtype] per row (in place). h [n x d], bias [3 x d]."
  [h bias qtype-ids n d]
  (add-qtype-bias* (:p h) (:p bias) (:p qtype-ids) (long n) (long d) (:p h))
  h)

(defn softmax
  "Row softmax over k columns."
  [x k]
  (let [n (quot (:size x) k)
        out (make (:shape x))]
    (softmax* (:p x) (long n) (long k) (:p out))
    out))

(defn add-scaled
  "out = a + alpha*b over an [n x d] block."
  [a b alpha]
  (let [n (long (first (:shape a)))
        d (long (if (second (:shape a)) (second (:shape a)) 1))
        out (make (:shape a))]
    (add-scaled* (:p a) (:p b) n d (float alpha) (:p out))
    out))

(defn relu!
  "In-place ReLU."
  [x]
  (relu* (:p x) (long (:size x)) (:p x))
  x)

(defn gelu!
  "In-place erf-GELU."
  [x]
  (gelu* (:p x) (long (:size x)) (:p x))
  x)

(defn reshape
  [t shape]
  (assoc t :shape (vec shape)))

(defn t-size [t] (:size t))

;; --- test helpers -------------------------------------------------------------

(defn eq-bytes?
  "Byte-wise equality of two uint8 'tensors' (allowed masks)."
  [a b]
  (and (= (:size a) (:size b))
       (loop [i 0]
         (or (= i (:size a))
             (and (= (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i))
                  (recur (inc i)))))))

(defn first-mismatch
  [a b]
  (loop [i 0]
    (if (or (= i (:size a)) (= (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)))
      (if (= i (:size a)) :none [i (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)])
      [i (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)])))

(defn get
  [p i]
  (ffi/read p :float (* 4 i)))
