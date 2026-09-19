(ns laya.tensors-test
  "Parity of the C kernels and the sgemm matmul against the torch oracle.

  golden/masks.edn  - exact allowed-attention matrices from transformers'
                      own mask builders (padded batch, lens [64 45])
  golden/layers/*   - f32 sidecars from dump_traces.py (ids, embeddings)
  data/model/*      - the checkpoint's own weights, upcast f32"
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [jolt.ffi]
            [laya.model :as laya.model]
            [laya.tensors :as t]))

(def golden-dir
  (or (System/getenv "LAYA_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LAYA_DATA") "data"))

(defn- slurp-edn [f]
  (edn/read-string (slurp f)))

(defn- read-golden-f32 [k edn]
  (let [{:keys [shape file]} (get edn k)]
    (t/load-file (str golden-dir "/" file) shape)))

(defn max-abs-diff [a b]
  (let [n (t/size a)
        na (t/ptr a)
        nb (t/ptr b)]
    (loop [i 0, m 0.0]
      (if (= i n)
        m
        (let [d (Math/abs (- (t/get na i) (t/get nb i)))]
          (recur (inc i) (max m d)))))))

(defn rel-l2 [a b]
  (let [n (t/size a)
        na (t/ptr a)
        nb (t/ptr b)]
    (loop [i 0, num 0.0, den 0.0]
      (if (= i n)
        (Math/sqrt (/ num (max den 1e-30)))
        (let [x (t/get na i)
              y (t/get nb i)
              d (- x y)]
          (recur (inc i) (+ num (* d d)) (+ den (* y y))))))))

;; --- the attention masks transformers itself builds -------------------------

(deftest mask-builder-matches-transformers
  (let [m (slurp-edn (str golden-dir "/masks.edn"))
        lens (vec (m :lens))
        L 64
        att (t/from-bytes (concat (repeat L 1)
                                  (repeat (lens 1) 1)
                                  (repeat (- L (lens 1)) 0))
                          [2 L])]
    (is (= [2 L] (t/shape att)) "att rows for the padded batch")
    (doseq [k [:full :sliding]]
      (let [gold (t/from-bytes (flatten (first (get m k))) [L L])
            got (t/allowed-mask att 1 L (if (= k :sliding) 64 -1))]
        (testing (str k " allowed matrix, padded batch row 1")
          (is (t/eq-bytes? gold got)
              (str "first mismatch " (t/first-mismatch gold got))))))))

;; --- embeddings: row gather + weight-only LayerNorm -------------------------

(deftest embeddings-match-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        emb-w (t/load-tensor manifest data-dir "encoder.embeddings.tok_embeddings.weight")
        norm-w (t/load-tensor manifest data-dir "encoder.embeddings.norm.weight")
        ids (map int (flatten (layers :ids)))
        n-tokens (count (flatten (layers :att)))
        got (t/embeddings emb-w (t/from-ints ids) n-tokens (cfg :hidden-size) norm-w)
        gold (read-golden-f32 :embeddings layers)]
    (is (< (max-abs-diff got gold) 1e-4)
        (str "embeddings max-abs " (max-abs-diff got gold)))))

;; --- matmul: sgemm vs a golden torch product ---------------------------------

(deftest sgemm-matches-torch-linear
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        gold (read-golden-f32 :matmul layers)
        X (read-golden-f32 :matmul-x layers)
        W (t/load-tensor manifest data-dir "scorer.1.weight")
        got (t/mmul X W)]
    (is (= (t/shape gold) (t/shape got)))
    (is (< (rel-l2 got gold) 1e-5)
        (str "sgemm rel-l2 " (rel-l2 got gold)))))

;; --- encoder: layer-by-layer vs golden ---------------------------------------

(deftest encoder-matches-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        w (laya.model/load-weights manifest data-dir)
        ids0 (t/from-ints (map int (first (layers :ids))))
        att0 (t/from-bytes (first (layers :att)) [74])
        gold-emb (read-golden-f32 :embeddings layers)
        emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                          ids0 74 1024 (w "encoder.embeddings.norm.weight"))
        full (t/allowed-mask att0 0 74 -1)
        slid (t/allowed-mask att0 0 74 64)
        mx (fn [a gold off n]
             (loop [i 0 r 0.0]
               (if (= i n) r
                   (recur (inc i)
                          (max r (Math/abs (- (t/get (t/ptr a) i)
                                              (t/get (t/ptr gold) (+ off i)))))))))]
    (is (< (mx emb gold-emb 0 (* 74 1024)) 1e-4)
        "embeddings row 0 within 1e-4 of torch")
    (doseq [i [0 1 2 27]]
      (let [out (reduce (fn [h k] (if (= k i)
                                    (reduced (laya.model/encoder-layer! w cfg k h att0 full slid))
                                    (laya.model/encoder-layer! w cfg k h att0 full slid)))
                        emb (range (inc i)))
            gold (read-golden-f32 (keyword (str "layer-" i)) layers)]
        (is (< (mx out gold 0 (* 74 1024)) (if (= i 27) 0.05 1e-4))
            (str "layer " i " max-abs " (mx out gold 0 (* 74 1024))))))
    (testing "padded batch row 1 (64 real tokens): real rows match, pad rows ignored"
      (let [ids1 (t/from-ints (map int (second (layers :ids))))
            att1 (t/from-bytes (second (layers :att)) [74])
            emb1 (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                               ids1 74 1024 (w "encoder.embeddings.norm.weight"))
            full1 (t/allowed-mask att1 0 74 -1)
            slid1 (t/allowed-mask att1 0 74 64)
            n-real (* 64 1024)
            off (* 74 1024)]
        (is (< (mx emb1 gold-emb off n-real) 1e-4) "embeddings row 1")
        (reduce (fn [h k]
                  (let [h2 (laya.model/encoder-layer! w cfg k h att1 full1 slid1)]
                    (when (#{0 1 2 27} k)
                      (let [gold (read-golden-f32 (keyword (str "layer-" k)) layers)]
                        (is (< (mx h2 gold off n-real) (if (= k 27) 0.05 1e-4))
                            (str "row 1 layer " k " max-abs " (mx h2 gold off n-real)))))
                    h2))
                emb1 (range 28))))))

;; --- head: TransformerEncoderLayers + scorer + ReLU discrimination ----------

(deftest head-matches-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        w (laya.model/load-weights manifest data-dir)
        att0 (t/from-bytes (first (layers :att)) [74])
        full (t/allowed-mask att0 0 74 -1)
        gold (fn [k shape] (t/load-file (str golden-dir "/layers/" k ".f32") shape))
        hin (let [g (gold "head-in" [2 74 1024])]
              {:p (t/ptr g) :shape [74 1024] :size (* 74 1024)})
        h0 (laya.model/head-layer! w cfg 0 hin full)
        h1 (laya.model/head-layer! w cfg 1 h0 full)
        mx (fn [a k shape]
             (let [g (gold k shape)]
               (loop [i 0 r 0.0]
                 (if (= i (* 74 1024)) r
                     (recur (inc i)
                            (max r (Math/abs (- (t/get (t/ptr a) i)
                                                (t/get (t/ptr g) i)))))))))]
    (is (< (mx h0 "head-layer-0" [2 74 1024]) 1e-3)
        (str "head layer 0 max-abs " (mx h0 "head-layer-0" [2 74 1024])))
    (is (< (mx h1 "head-layer-1" [2 74 1024]) 1e-3)
        (str "head layer 1 max-abs " (mx h1 "head-layer-1" [2 74 1024])))
    (let [markers (t/reshape (gold "markers" [2 4 1024]) [8 1024])
          logits (laya.model/scorer w markers)
          gold-logits (gold "logits" [2 4])
          ldiff (loop [i 0 r 0.0]
                  (if (= i 4) r
                      (recur (inc i)
                             (max r (Math/abs (- (t/get (t/ptr logits) i)
                                                 (t/get (t/ptr gold-logits) i)))))))
          gelu-logits (gold "gelu-logits" [2 4])
          ggap (loop [i 0 r 0.0]
                 (if (= i 4) r
                     (recur (inc i)
                            (max r (Math/abs (- (t/get (t/ptr logits) i)
                                                (t/get (t/ptr gelu-logits) i)))))))]
      (is (< ldiff 1e-3) (str "logits max-abs " ldiff))
      (is (> ggap (* 0.9 (layers :relu-vs-gelu-gap)))
          "jolt head is ReLU: it must sit a full gap away from the GELU trace"))
    (testing "padded row 1 through the head with its key-padding mask"
      (let [att1 (t/from-bytes (second (layers :att)) [74])
            full1 (t/allowed-mask att1 0 74 -1)
            g (gold "head-in" [2 74 1024])
            hin1 {:p (jolt.ffi/segment (+ (jolt.ffi/address (t/ptr g)) (* 74 1024 4)))
                  :shape [74 1024] :size (* 74 1024)}
            h1 (laya.model/head-layer! w cfg 1 (laya.model/head-layer! w cfg 0 hin1 full1) full1)
            gold1 (gold "head-layer-1" [2 74 1024])
            mx1 (loop [i 0 r 0.0]
                  (if (= i (* 64 1024)) r
                      (recur (inc i)
                             (max r (Math/abs (- (t/get (t/ptr h1) i)
                                                 (t/get (t/ptr gold1) (+ (* 74 1024) i))))))))
            m1 (t/gather h1 (t/from-ints (take 2 (second (layers :marker-pos)))) 2 1024)
            lg1 (t/to-floats (laya.model/scorer w m1))
            gold-logits (gold "logits" [2 4])]
        (is (< mx1 1e-3) (str "row 1 head layer 1 max-abs " mx1))
        (is (< (Math/abs (- (first lg1) (t/get (t/ptr gold-logits) 4))) 1e-3))
        (is (< (Math/abs (- (second lg1) (t/get (t/ptr gold-logits) 5))) 1e-3))))))

;; --- act-head features: top1, top1-top2, normalized entropy, k/255 ---------

(deftest act-features-match-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        gold-logits (t/load-file (str golden-dir "/layers/logits.f32") [2 4])
        feats (layers :act-feats)]
    (doseq [r [0 1]]
      (let [mask (nth (layers :marker-mask) r)
            k (count (filter pos? mask))
            ;; per-row forward has no padded markers: only the k real logits
            lg (mapv #(t/get (t/ptr gold-logits) (+ (* r 4) %)) (range k))
            got (laya.model/answer-features (t/from-floats lg) (repeat k 1))
            want (nth feats r)]
        (doseq [[i name] [[0 "top1"] [1 "top1-top2"] [2 "entropy"] [3 "k/255"]]]
          (is (< (Math/abs (- (double (nth got i)) (double (nth want i)))) 1e-5)
              (str "row " r " " name " got " (nth got i) " want " (nth want i))))))))

(deftest top2-of-any-order
  (is (= [0.5 0.3] (laya.model/top2 [0.5 0.3 0.1 0.1])) "descending input")
  (is (= [0.5 0.3] (laya.model/top2 [0.1 0.3 0.5])) "ascending input")
  (is (= [0.5 0.3] (laya.model/top2 [0.3 0.5 0.1])) "mixed")
  (is (= [0.4 0.4] (laya.model/top2 [0.4 0.2 0.4])) "ties keep both"))
