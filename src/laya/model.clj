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
        ;; cached across calls, so not owned by the per-row arena
        (let [ts (binding [t/*arena* nil] (t/rope-tables theta (:head-dim cfg) L))]
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

;; --- workspace -----------------------------------------------------------------

(defn workspace
  "Every intermediate a layer needs for B rows of L tokens, allocated once
  so the 28 encoder layers and the 2 head layers overwrite the same buffers
  instead of each taking ~45 MB of fresh (zeroed, page-faulted) memory per
  layer, which cost ~7 ms a layer at L=512, a fifth of the forward. The
  rows are stacked, [B*L x d], so every gemm reads the weights once for
  the whole batch; only attention goes row by row and its buffers are per
  row. Each kernel writes every element of its output, so nothing needs
  zeroing between uses. The residual stream alternates between :res0 and
  :res1 so a layer never writes into the tensor it was given
  (`next-residual`)."
  [cfg B L]
  (let [n (* B L)
        d (:hidden-size cfg)
        H (:num-heads cfg)
        hd (:head-dim cfg)
        mid (:intermediate cfg)
        ff (max (* 2 mid) (* 4 d))]   ; encoder Wi [n x 2mid], head linear1 [n x 4d]
    {:B B :L L
     :res0 (t/make [n d]) :res1 (t/make [n d])
     :norm (t/make [n d]) :qkv (t/make [n (* 3 d)])
     :qh (t/make [(* H L) hd]) :kh (t/make [(* H L) hd]) :vh (t/make [(* H L) hd])
     :ctx (t/make [n d]) :proj (t/make [n d]) :h2 (t/make [n d])
     :ff (t/make [n ff]) :sw (t/make [n mid]) :mo (t/make [n d])
     :S (t/make [L L]) :P (t/make [L L])}))

(defn- next-residual
  "The residual buffer that is not h: h came from the other one, or from
  outside the workspace (the embeddings), in which case either will do."
  [ws h]
  (if (= (ffi/address (t/ptr h)) (ffi/address (t/ptr (:res0 ws)))) (:res1 ws) (:res0 ws)))

(defn- attention-rows!
  "Per-row attention over the stacked qkv [B*L x 3d]: split each row's
  heads, optionally rope them, and write its context into its rows of
  the workspace's :ctx. `allowed-for` gives row b's [L x L] mask."
  [cfg qkv allowed-for rope window ws]
  (let [{:keys [B L qh kh vh ctx S P]} ws
        H (:num-heads cfg)
        hd (:head-dim cfg)]
    (dotimes [b B]
      (split-qkv* (t/ptr (t/rows qkv (* b L) L)) (long L) (long H) (long hd)
                  (t/ptr qh) (t/ptr kh) (t/ptr vh))
      (when-let [[cos-t sin-t] rope]
        (t/rope-apply! qh cos-t sin-t H L hd)
        (t/rope-apply! kh cos-t sin-t H L hd))
      (t/attention! (t/rows ctx (* b L) L) S P qh kh vh (allowed-for b)
                    H L hd (/ 1.0 (Math/sqrt hd)) window))
    ctx))

(defn encoder-layer!
  "Run encoder layer i on the stacked rows h [B*L x d]; masks is one
  [full sliding] pair of [L x L] byte masks per row. h is read, not
  written: the result is the workspace residual buffer that is not h (or a
  fresh tensor without a workspace)."
  ([w cfg i h masks]
   (encoder-layer! w cfg i h masks
                   (workspace cfg (count masks) (quot (first (t/shape h)) (count masks)))))
  ([w cfg i h masks ws]
   (let [n (first (t/shape h))
         L (:L ws)
         sliding? (= "sliding_attention" (nth (:layer-types cfg) i))
         attn-in (if (zero? i)
                   h
                   (t/layernorm! (:norm ws) h (w (wname i "attn_norm.weight")) (:norm-eps cfg)))
         qkv (t/mmul! (:qkv ws) attn-in (w (wname i "attn.Wqkv.weight")))
         ctx (attention-rows! cfg qkv
                              (fn [b] (nth (nth masks b) (if sliding? 1 0)))
                              (rope-tables-for cfg i L)
                              (if sliding? (:window cfg) -1)
                              ws)
         attn-out (t/mmul! (:proj ws) ctx (w (wname i "attn.Wo.weight")))
         h2 (t/add-scaled! (:h2 ws) h attn-out 1.0)
         mlp-in (t/layernorm! (:norm ws) h2 (w (wname i "mlp_norm.weight")) (:norm-eps cfg))
         mid (:intermediate cfg)
         wi (t/mmul! (t/reshape (:ff ws) [n (* 2 mid)]) mlp-in (w (wname i "mlp.Wi.weight")))
         sw (t/swiglu! (:sw ws) wi mid)
         mo (t/mmul! (:mo ws) sw (w (wname i "mlp.Wo.weight")))]
     (t/add-scaled! (next-residual ws h) h2 mo 1.0))))

(defn row-masks
  "The [full sliding] mask pair of every row of att [B x L]."
  [cfg att B L]
  (vec (for [b (range B)]
         [(t/allowed-mask att b L -1) (t/allowed-mask att b L (:window cfg))])))

(defn encode-batch
  "Full encoder for B stacked rows: embeddings, 28 layers, final norm.
  ids [B*L] token ids (rows padded to L), att [B x L] bytes 1=present.
  The workspace is shared with the head layers that follow."
  [w cfg ids-tensor att masks ws]
  (let [{:keys [B L]} ws
        d (:hidden-size cfg)
        emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                          ids-tensor (* B L) d
                          (w "encoder.embeddings.norm.weight"))
        h (reduce (fn [h i] (encoder-layer! w cfg i h masks ws))
                  emb
                  (range (:num-layers cfg)))]
    ;; into the residual buffer h did not come from: the head reads it next
    (t/layernorm! (next-residual ws h) h (w "encoder.final_norm.weight") (:norm-eps cfg))))

;; --- head ---------------------------------------------------------------------

(defn top2
  "Two largest values of a float seq, largest first (torch topk(2).values)."
  [xs]
  (reduce (fn [[a b] x] (cond (> x a) [x a] (> x b) [a x] :else [a b]))
          [##-Inf ##-Inf]
          xs))

(defn head-attention
  "torch MultiheadAttention (in_proj/out_proj, biased) with key-padding
  mask, on the stacked rows x [B*L x d] -> [B*L x d] (the workspace's
  :proj); alloweds is one [L x L] padding mask per row. Same math as the
  encoder block but biased, without rope and full-mask-only."
  [w prefix x alloweds cfg ws]
  (let [n (first (t/shape x))
        d (:hidden-size cfg)
        qkv (t/mmul! (:qkv ws) x (w (str prefix ".self_attn.in_proj_weight")))
        _ (add-bias! (t/ptr qkv) (t/ptr (w (str prefix ".self_attn.in_proj_bias")))
                     (long n) (long (* 3 d)))
        ctx (attention-rows! cfg qkv #(nth alloweds %) nil -1 ws)
        out (t/mmul! (:proj ws) ctx (w (str prefix ".self_attn.out_proj.weight")))]
    (add-bias! (t/ptr out) (t/ptr (w (str prefix ".self_attn.out_proj.bias")))
               (long n) (long d))
    out))

(defn head-layer!
  "One torch TransformerEncoderLayer, norm_first=true, ReLU FF, biased LN,
  on the stacked rows x [B*L x d]; alloweds is one [L x L] padding mask
  per row. x is read, not written, as in encoder-layer!."
  ([w cfg li x alloweds]
   (head-layer! w cfg li x alloweds
                (workspace cfg (count alloweds) (quot (first (t/shape x)) (count alloweds)))))
  ([w cfg li x alloweds ws]
   (let [n (first (t/shape x))
         d (:hidden-size cfg)
         prefix (str "head.layers." li)
         n1 (t/layernorm! (:norm ws) x (w (str prefix ".norm1.weight"))
                          (w (str prefix ".norm1.bias")) 1e-5)
         attn (head-attention w prefix n1 alloweds cfg ws)
         x2 (t/add-scaled! (:h2 ws) x attn 1.0)
         n2 (t/layernorm! (:norm ws) x2 (w (str prefix ".norm2.weight"))
                          (w (str prefix ".norm2.bias")) 1e-5)
         l1 (t/mmul! (t/reshape (:ff ws) [n (* 4 d)]) n2 (w (str prefix ".linear1.weight")))
         _ (add-bias! (t/ptr l1) (t/ptr (w (str prefix ".linear1.bias")))
                      (long n) (long (* 4 d)))
         _ (t/relu! l1)
         l2 (t/mmul! (:mo ws) l1 (w (str prefix ".linear2.weight")))
         _ (add-bias! (t/ptr l2) (t/ptr (w (str prefix ".linear2.bias")))
                      (long n) (long d))]
     (t/add-scaled! (next-residual ws x) x2 l2 1.0))))

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
        ent (/ (- (reduce + (map (fn [p] (if (pos? p) (* p (Math/log (max p 1e-9))) 0.0)) ps)))
               (Math/log k))
        top1 (if (= t1 ##-Inf) 0.0 t1)
        top2v (if (= t2 ##-Inf) 0.0 t2)]
    [top1 (- top1 top2v) ent (/ k 255.0)]))

(defn forward-batch
  "B rows through encoder + decision head in one pass: every gemm runs on
  the rows stacked [B*L x d] (the weights are read once per layer for the
  batch, and a gemm on more rows runs closer to peak), attention runs per
  row. Rows are padded to the longest with the pad id and masked out, so
  a row's real tokens come out as if it ran alone. Each row is
  {:ids [L_b] :att [L_b] :markers [k_b] :marker-mask [k_b] :qtype q};
  answers [[logits act-logits] ...] as float vectors, in row order. Every
  intermediate tensor is owned by an arena that closes on return."
  [w cfg rows]
  (with-open [arena (ffi/confined-arena)]
    (binding [t/*arena* arena]
      (let [d (:hidden-size cfg)
            B (count rows)
            L (reduce max (map #(count (:ids %)) rows))
            pad (or (:pad-id cfg) 0)
            padded (fn [xs fill] (into (vec xs) (repeat (- L (count xs)) fill)))
            ws (workspace cfg B L)
            ids (t/from-ints (mapcat #(padded (:ids %) pad) rows))
            att (t/from-bytes (mapcat #(padded (:att %) 0) rows) [B L])
            masks (row-masks cfg att B L)
            h (encode-batch w cfg ids att masks ws)
            _ (t/add-qtype! h (w "type_emb.weight")
                            (t/from-ints (mapcat #(repeat L (:qtype %)) rows)) (* B L) d)
            h2 (reduce (fn [hh li] (head-layer! w cfg li hh (mapv first masks) ws))
                       h (range (:head-layers cfg)))
            ;; every row's markers through the scorer at once
            kpos (vec (mapcat (fn [b {:keys [markers]}]
                                (map #(+ (* b L) (max 0 (long %))) markers))
                              (range) rows))
            lg (t/to-floats (scorer w (t/gather h2 (t/from-ints kpos) (count kpos) d)))]
        (loop [b 0, at 0, out []]
          (if (= b B)
            out
            (let [{:keys [markers marker-mask]} (nth rows b)
                  k (count markers)
                  lgv (mapv (fn [v msk] (if (pos? (long msk)) v -1e4))
                            (subvec lg at (+ at k)) marker-mask)
                  feats (answer-features (t/from-floats lgv) marker-mask)
                  pooled (t/gather h2 (t/from-ints [(* b L)]) 1 d)]
              (recur (inc b) (+ at k)
                     (conj out [lgv (t/to-floats (act-head w pooled feats))])))))))))

(defn forward-row
  "One row through encoder + decision head: forward-batch of a batch of one.
  ids-row/att-row are [L] seqs; marker-pos/marker-mask are [kmax] seqs.
  Returns [logits act-logits] as float vectors."
  [w cfg ids-row att-row marker-pos marker-mask qtype]
  (first (forward-batch w cfg [{:ids ids-row :att att-row :markers marker-pos
                                :marker-mask marker-mask :qtype qtype}])))
