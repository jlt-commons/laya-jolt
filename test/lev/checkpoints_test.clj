(ns lev.checkpoints-test
  "Parity of the other checkpoints in the bundle, each against its own
  golden/<name>/ dump from the torch oracle and its own data/<name>/ from
  jolt prepare: config, tokenizer cases, build-sequence branches under the
  checkpoint's max_len / head_max_len, the last encoder layer, and the
  quickstart + typed-decisions answers through the Router."
  (:require [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.model]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.tensors :as t]
            [lev.test-util :as tu]
            [lev.tokenizer :as tk]))

(def checkpoints
  "Every golden/<name>/ that exists (each needs data/<name>/ prepared), or
  the names in LEV_CHECKPOINTS (comma-separated; empty means none), so CI
  can test one checkpoint per process."
  (let [wanted (some-> (System/getenv "LEV_CHECKPOINTS") (clojure.string/split #",") set)]
    (vec (for [n ["typed-decisions" "multilingual"]
               :when (.exists (io/file tu/golden-dir n "cases.edn"))
               :when (or (nil? wanted) (contains? wanted n))]
           n))))

(def agents
  (into {} (for [n checkpoints]
             [n (delay (ag/load-agent (str tu/data-dir "/" n)))])))

(defn- golden [n file]
  (edn/read-string {:readers tu/readers} (slurp (str tu/golden-dir "/" n "/" file ".edn"))))

(defn- agent [n]
  (is (.exists (io/file tu/data-dir n "manifest.edn"))
      (str "data/" n " is not prepared: jolt -M:prepare --model " n))
  @(get agents n))

(deftest checkpoints-are-prepared
  (when (nil? (System/getenv "LEV_CHECKPOINTS"))
    (is (= ["typed-decisions" "multilingual"] checkpoints) "both extra checkpoints have goldens"))
  (doseq [n checkpoints]
    (is (.exists (io/file tu/data-dir n "manifest.edn")) (str n " prepared"))))

(deftest config-and-tokenizer
  (doseq [n checkpoints]
    (let [{:keys [cfg tok]} (agent n)
          cases (golden n "cases")
          tok-golden (:cases (golden n "tok"))]
      (testing (str n ": config")
        (is (= 1024 (:max-len cfg)))
        (is (= 256 (:head-max-len cfg))))
      (testing (str n ": specials the oracle used")
        (let [sp (:specials cases)]
          (is (= (get sp "cls") (:cls (:specials tok))))
          (is (= (get sp "sep") (:sep (:specials tok))))
          (is (= (get sp "mask") (:mask (:specials tok))))
          (is (= (get sp "pad") (:pad (:specials tok))))))
      (testing (str n ": tokenizer cases")
        (doseq [[i text] (map-indexed vector (:tok-cases cases))]
          (is (= (mapv long (get tok-golden (str i))) (tk/encode tok text)) (str "case " i)))))))

(deftest sequences-under-the-checkpoints-limits
  (doseq [n checkpoints]
    (let [{:keys [cfg tok]} (agent n)
          cases (golden n "sequences")]
      (is (= 17 (count cases)))
      (doseq [[name {:keys [state question ids markers options]}] cases]
        (let [q (ag/to-internal question)
              [got-ids got-markers] (seq/build-sequence tok state q (:max-len cfg) (:head-max-len cfg))]
          (is (= options (seq/render-options q)) (str n " " name))
          (is (= (mapv long ids) got-ids) (str n " " name))
          (is (= (mapv long markers) got-markers) (str n " " name)))))))

(deftest last-encoder-layer-matches-torch
  (doseq [n checkpoints]
    (let [{:keys [cfg w]} (agent n)
          layers (golden n "layers")
          L (:L layers)
          d (:hidden layers)
          last (dec (:num-layers layers))
          ids0 (t/from-ints (map int (first (:ids layers))))
          att0 (t/from-bytes (first (:att layers)) [L])
          emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                            ids0 L d (w "encoder.embeddings.norm.weight"))
          full (t/allowed-mask att0 0 L -1)
          slid (t/allowed-mask att0 0 L (:window cfg))
          load (fn [k] (let [{:keys [shape file]} (get layers k)]
                         (t/load-file (str tu/golden-dir "/" n "/" file) shape)))
          mx (fn [a gold nvals]
               (loop [i 0 r 0.0]
                 (if (= i nvals) r
                     (recur (inc i) (max r (Math/abs (- (t/get (t/ptr a) i) (t/get (t/ptr gold) i))))))))]
      (testing (str n ": embeddings and layer 0 at f32 noise, layer " last " within the residual-stream bound")
        (is (= d (:hidden-size cfg)))
        (is (= (:num-layers layers) (:num-layers cfg)))
        (is (< (mx emb (load :embeddings) (* L d)) 1e-4))
        (let [out (reduce (fn [h k] (lev.model/encoder-layer! w cfg k h [[full slid]])) emb (range (inc last)))
              l0 (lev.model/encoder-layer! w cfg 0 emb [[full slid]])
              [rel diff scale] (tu/relative-max-abs out (load (keyword (str "layer-" last))) (* L d))]
          (is (< (mx l0 (load :layer-0) (* L d)) 1e-4) "layer 0")
          ;; same bound as tensors_test/last-layer-rel-tol
          (is (< rel 5e-5) (format "layer %d: max-abs %.4g of a %.4g-scale stream (rel %.2e)" last diff scale rel)))))))

(deftest answers-match-the-oracle
  (doseq [n checkpoints]
    (let [ag* (agent n)
          readme (golden n "readme")
          cases (golden n "cases")
          rt (router/make-router {:models {n (str tu/data-dir "/" n)}
                                  :loader (fn [name & _] (if (= name n) ag* (throw (ex-info "wrong checkpoint" {:name name}))))})]
      (testing (str n ": the quickstart (to the fourth decimal)")
        (tu/answers-match (:system-one readme)
                          (seq/json-str (ag/system-one ag* (:readme-state cases) (:readme-questions cases)))))
      (testing (str n ": an invoice-processing case, routed explicitly")
        (let [out (router/predict rt (:typed-decisions-state readme) (:typed-decisions-questions readme) :model n)
              want (json/read-str (:typed-decisions readme))
              got (json/read-str (seq/json-str (dissoc out "routing")))]
          (is (= n (get-in out ["routing" "model"])))
          (is (= (get want "usage") (get got "usage")))
          (is (= (into {} (map (fn [[k a]] [k (get a "choice")]) (get want "answers")))
                 (into {} (map (fn [[k a]] [k (get a "choice")]) (get got "answers")))))
          ;; float32 softmax in numpy vs doubles here: one unit in the last place
          (is (tu/approx= 1.0001e-4 want got))))
      (testing (str n ": email fan-out")
        (let [g (golden n "email_answers")
              bodies (mapv first (:clean (tu/read-golden "email")))]
          (doseq [{:keys [body-index state result]} (:cases g)]
            (let [want (json/read-str result)
                  got (json/read-str (seq/json-str (ag/system-one ag* state (:questions g))))]
              (is (tu/approx= 1.0001e-4 want got) (str n " body " body-index)))))))))
