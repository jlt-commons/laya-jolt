(ns laya.model
  "ModernBERT-large encoder + the Laya decision head, f32, verified layer by
  layer against the torch golden traces in golden/layers.edn.

  Encoder layer (modeling_modernbert.py; norm_bias=attention_bias=false):
    h = h + attn(attn_norm(h))      ; attn_norm = Identity on layer 0
    h = h + mlp(mlp_norm(h))        ; mlp = Wo(swiglu(Wi h)), Wi -> 2*2624
  Attention: Wqkv, per-head RoPE (theta 160k full / 10k sliding, hd=64),
    scores * hd^-0.5, padding+window mask (|i-j|<=64 on sliding layers),
    softmax f32, @V, Wo. Full attention every 3rd layer from 0.

  Head (torch nn.TransformerEncoderLayer x2: norm_first, batch_first,
  nhead=16, ff=4096, ReLU — verified against torch source AND the golden
  gelu-logits gap):
    h += type_emb[qtype]; marker gather; scorer = LN-Linear-GELU-Linear;
    act head on pooled[0] + [top1, top1-top2, norm-ent, k/255]."
  (:require [jolt.ffi :as ffi]
            [laya.tensors :as t]))

;; --- extra kernel bindings ----------------------------------------------------

(ffi/defcfn split-qkv* "lla_split_qkv"
  [:pointer :int64 :int64 :int64 :pointer :pointer :pointer] :void)
(ffi/defcfn attention* "lla_attention"
  [:pointer :pointer :pointer :pointer :int64 :int64 :int64 :double :pointer] :void)
(ffi/defcfn add-bias! "lla_add_bias" [:pointer :pointer :int64 :int64] :void)

(def ^:private rope-cache (atom {}))

(defn rope-tables-for
  "cos/sin for this layer's theta (full 160k / sliding 10k), cached per L."
  [cfg i L]
  (let [theta (if (= "full_attention" (nth (:layer-types cfg) i))
                (:rope-full cfg)
                (:rope-local cfg))
        k [theta L]]
    (or (get @rope-cache k)
        (let [ts (t/rope-tables theta (:head-dim cfg) L)]
          (swap! rope-cache assoc k ts)
          ts))))

(defn load-weights
  "Load every tensor in the manifest under data-dir; answers {name tensor}."
  [manifest data-dir]
  (into {}
        (map (fn [[name {:keys [shape file]}]]
               [name (t/load-file (str data-dir "/" file) shape)]))
        (:tensors manifest)))

(defn- wname [i k] (str "encoder.layers." i "." k))

(declare encoder-attention)

(defn encoder-layer!
  "Run encoder layer i on ONE batch row. h [L x d] in-place semantics:
  returns a NEW tensor. allowed-full/allowed-sliding are [L x L] byte masks."
  [w cfg i h att-row allowed-full allowed-sliding]
  (let [L (first (t/shape h))
        d (:hidden-size cfg)
        H (:num-heads cfg)
        hd (:head-dim cfg)
        sliding? (= "sliding_attention" (nth (:layer-types cfg) i))
        allowed (if sliding? allowed-sliding allowed-full)
        attn-in (if (zero? i)
                  h
                  (t/layernorm h (w (wname i "attn_norm.weight")) (:norm-eps cfg)))
        ctx (encoder-attention w cfg i attn-in allowed L)
        attn-out (t/mmul ctx (w (wname i "attn.Wo.weight")))
        h2 (t/add-scaled h attn-out 1.0)
        mlp-in (t/layernorm h2 (w (wname i "mlp_norm.weight")) (:norm-eps cfg))
        wi (t/mmul mlp-in (w (wname i "mlp.Wi.weight")))
        mid (:intermediate cfg)
        sw (t/swiglu wi mid)
        mo (t/mmul sw (w (wname i "mlp.Wo.weight")))]
    (t/add-scaled h2 mo 1.0)))

(defn encoder-attention
  "qkv-split, rope, masked attention, Wo. One batch row, [L x d] in and out."
  [w cfg i attn-in allowed L]
  (let [d (:hidden-size cfg)
        H (:num-heads cfg)
        hd (:head-dim cfg)
        qkv (t/mmul attn-in (w (wname i "attn.Wqkv.weight")))
        qh (t/make [(* H L) hd])
        kh (t/make [(* H L) hd])
        vh (t/make [(* H L) hd])
        _ (split-qkv* (t/ptr qkv) (long L) (long H) (long hd)
                      (t/ptr qh) (t/ptr kh) (t/ptr vh))
        [cos-t sin-t] (rope-tables-for cfg i L)
        _ (t/rope-apply! qh cos-t sin-t H L hd)
        _ (t/rope-apply! kh cos-t sin-t H L hd)
        ctx (t/make [L d])]
    (attention* (t/ptr qh) (t/ptr kh) (t/ptr vh) (t/ptr allowed)
                (long H) (long L) (long hd)
                (/ 1.0 (Math/sqrt hd))
                (t/ptr ctx))
    ctx))

(defn encode-row
  "Full encoder for one batch row: embeddings, 28 layers, final norm.
  ids [L] token ids, att-row [L] bytes 1=present."
  [w cfg ids-tensor att-row]
  (let [L (t/size att-row)
        d (:hidden-size cfg)
        emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                          ids-tensor L d
                          (w "encoder.embeddings.norm.weight"))
        full (t/allowed-mask att-row 0 L -1)
        sliding (t/allowed-mask att-row 0 L (:window cfg))
        h (reduce (fn [h i] (encoder-layer! w cfg i h att-row full sliding))
                  emb
                  (range (:num-layers cfg)))]
    (t/layernorm h (w "encoder.final_norm.weight") (:norm-eps cfg))))

;; --- head ---------------------------------------------------------------------

(defn- top2
  "Two largest values of a float seq, largest first."
  [xs]
  (reduce (fn [[a b] x] (cond (> x a) [x a] (> b x) [a x] :else [a b]))
          [##-Inf ##-Inf]
          xs))

(defn head-attention
  "torch MultiheadAttention (in_proj/out_proj, biased) with key-padding mask.
  x [L x d] -> [L x d]. Same math as the encoder block but biased and
  full-mask-only."
  [w prefix x allowed L cfg]
  (let [d (:hidden-size cfg)
        H (:num-heads cfg)
        hd (:head-dim cfg)
        qkv (t/mmul x (w (str prefix ".self_attn.in_proj_weight")))
        _ (add-bias! (t/ptr qkv) (t/ptr (w (str prefix ".self_attn.in_proj_bias")))
                     (long L) (long (* 3 d)))
        qh (t/make [(* H L) hd])
        kh (t/make [(* H L) hd])
        vh (t/make [(* H L) hd])
        _ (split-qkv* (t/ptr qkv) (long L) (long H) (long hd)
                      (t/ptr qh) (t/ptr kh) (t/ptr vh))
        ;; no rope in the head; straight scores
        ctx (t/make [L d])
        _ (attention* (t/ptr qh) (t/ptr kh) (t/ptr vh) (t/ptr allowed)
                      (long H) (long L) (long hd)
                      (/ 1.0 (Math/sqrt hd))
                      (t/ptr ctx))
        out (t/mmul ctx (w (str prefix ".self_attn.out_proj.weight")))]
    (add-bias! (t/ptr out) (t/ptr (w (str prefix ".self_attn.out_proj.bias")))
               (long L) (long d))
    out))

(defn head-layer!
  "One torch TransformerEncoderLayer, norm_first=true, ReLU FF, biased LN.
  x [L x d], allowed [L x L] padding mask."
  [w cfg li x allowed]
  (let [L (first (t/shape x))
        d (:hidden-size cfg)
        prefix (str "head.layers." li)
        n1 (t/layernorm x (w (str prefix ".norm1.weight"))
                        (w (str prefix ".norm1.bias")) 1e-5)
        attn (head-attention w prefix n1 allowed L cfg)
        x2 (t/add-scaled x attn 1.0)
        n2 (t/layernorm x2 (w (str prefix ".norm2.weight"))
                        (w (str prefix ".norm2.bias")) 1e-5)
        l1 (t/mmul n2 (w (str prefix ".linear1.weight")))
        _ (add-bias! (t/ptr l1) (t/ptr (w (str prefix ".linear1.bias")))
                     (long L) (long (* 4 d)))
        _ (t/relu! l1)
        l2 (t/mmul l1 (w (str prefix ".linear2.weight")))
        _ (add-bias! (t/ptr l2) (t/ptr (w (str prefix ".linear2.bias")))
                     (long L) (long d))]
    (t/add-scaled x2 l2 1.0)))

(defn scorer
  "LN(bias) -> Linear+GELU -> Linear(+bias), on [n x d] marker rows.
  Answers [n] logits."
  [w x]
  (let [[n d] (t/shape x)
        a (t/layernorm x (w "scorer.0.weight") (w "scorer.0.bias") 1e-5)
        b (t/mmul a (w "scorer.1.weight"))
        _ (add-bias! (t/ptr b) (t/ptr (w "scorer.1.bias")) (long n) (long d))
        _ (t/gelu! b)
        c (t/mmul b (w "scorer.3.weight"))]
    (add-bias! (t/ptr c) (t/ptr (w "scorer.3.bias")) (long n) 1)
    ;; c is [n x 1]; flatten
    (t/reshape c [n])))

(defn act-head
  "act logits from pooled row [d] + 4 features. Answers [2]."
  [w pooled feats]
  (let [xf (t/from-floats (concat (t/to-floats pooled) feats)) ; [d+4]
        xin (t/reshape xf [1 (t/size xf)])                      ; [1 x d+4]
        l0 (t/mmul xin (w "act_head.0.weight"))                 ; [1 x 256]
        _ (add-bias! (t/ptr l0) (t/ptr (w "act_head.0.bias")) 1 256)
        _ (t/gelu! l0)
        l2 (t/mmul l0 (w "act_head.2.weight"))                  ; [1 x 2]
        _ (add-bias! (t/ptr l2) (t/ptr (w "act_head.2.bias")) 1 2)]
    (t/reshape l2 [2])))

(defn answer-features
  "torch: p = softmax(logits); top2; ent = -(p log p)/log(k); k = count.
  logits [kmax] with -1e4 at invalid markers; marker-mask [kmax] 0/1."
  [logits-t marker-mask]
  (let [ps (t/to-floats (t/softmax logits-t (t/size logits-t)))
        mask (vec marker-mask)
        k (max 2 (count (filter pos? mask)))
        [t1 t2] (top2 ps)
        ent (->> ps
                 (map (fn [p] (if (pos? p) (* p (Math/log (max p 1e-9))) 0.0)))
                 (reduce +)
                 (- )
                 (/ (Math/log k)))
        top1 (if (= t1 ##-Inf) 0.0 t1)
        top2v (if (= t2 ##-Inf) 0.0 t2)]
    [top1 (- top1 top2v) ent (/ k 255.0)]))

(defn forward-row
  "One batch row through encoder + decision head.
  ids-row/att-row are [L] seqs; marker-pos/marker-mask are [kmax] seqs.
  Returns [logits act-logits] as float vectors."
  [w cfg ids-row att-row marker-pos marker-mask qtype]
  (let [d (:hidden-size cfg)
        L (count ids-row)
        att-t (t/from-bytes att-row)
        h (encode-row w cfg (t/from-ints ids-row) att-t)
        _ (t/add-qtype! h (w "type_emb.weight")
                        (t/from-ints (vec (repeat L qtype))) L d)
        allowed (t/allowed-mask att-t 0 L -1)
        h2 (reduce (fn [hh li] (head-layer! w cfg li hh allowed))
                   h (range (:head-layers cfg)))
        kmax (count marker-pos)
        kpos (mapv #(max 0 (long %)) marker-pos)
        m (t/gather h2 (t/from-ints kpos) kmax d)
        lg (t/to-floats (scorer w m))
        lgv (mapv (fn [v msk] (if (pos? (long msk)) v -1e4)) lg marker-mask)
        feats (answer-features (t/from-floats lgv) marker-mask)
        pooled (t/gather h2 (t/from-ints [0]) 1 d)]
    [lgv (t/to-floats (act-head w pooled feats))]))
