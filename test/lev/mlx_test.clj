(ns lev.mlx-test
  "lev.mlx: the encoder forward on Apple's GPU behind native/lev_mlx.c
  (jolt mlx). Without the native only the availability contract is
  checked; with it, the MLX forward is held to the same golden traces as
  the C kernels: the oracle's answers to the fourth decimal at f32, and
  the C kernels' own logits on the golden batch. f16 is the speed mode:
  argmax-exact on the goldens, values off by up to 1e-2."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.json :as json]
            [lev.mlx :as mlx]
            [lev.model :as m]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.tensors :as t]
            [lev.test-util :as tu :refer [golden-dir data-dir]]))

(def f32 (delay (mlx/load-agent data-dir {:name "english"})))
(def f16 (delay (mlx/load-agent data-dir {:name "english" :dtype :f16})))

(def cases (delay (tu/read-golden "cases")))

(deftest availability-is-a-question-not-a-crash
  (is (boolean? (mlx/available?)))
  (when-not (mlx/available?)
    (is (thrown-with-msg? Exception #"jolt mlx" (mlx/load-agent data-dir {})))
    (is (thrown-with-msg? Exception #"jolt mlx" (router/load-prepared "english" data-dir {:backend :mlx})))))

(deftest the-agent-says-what-it-is
  (when (mlx/available?)
    (let [a @f32]
      (is (= :encoder (:kind a)))
      (is (= :mlx (:backend a)))
      (is (= :f32 (:dtype a)))
      (is (= "english" (:name a)))
      (is (fn? (:forward a)))
      (is (fn? (:close a)))
      (is (= #{:p :gpu? :dtype :selected-head} (set (keys (:w a)))) ":w is the device handle, not tensors in the jolt heap")
      (is (:gpu? a) "on the GPU")
      (is (str/starts-with? (mlx/version) "mlx-c"))
      (is (= :f16 (:dtype @f16))))))

(deftest a-missing-tensor-file-is-an-error-with-its-name
  (when (mlx/available?)
    ;; a prepared directory's config, manifest and tokenizer without its model/
    (let [dir "target/mlx-partial"]
      (clojure.java.io/make-parents (str dir "/x"))
      (doseq [f ["manifest.edn" "config.edn" "tokenizer.edn"]]
        (spit (str dir "/" f) (slurp (str data-dir "/" f))))
      (let [e (try (mlx/load-agent dir {}) nil (catch Exception e e))]
        (is (some? e))
        (is (= :model-unavailable (:type (ex-data e))))
        (is (re-find #"cannot read \d+ floats from target/mlx-partial/model/[a-z_.0-9]+\.f32" (ex-message e))
            (str "names the tensor file it could not read: " (ex-message e)))))))

(deftest mlx-forward-matches-the-c-kernels-on-the-golden-batch
  (when (mlx/available?)
    (let [layers (edn/read-string {:readers tu/readers} (slurp (str golden-dir "/layers.edn")))
          cfg (:cfg @tu/agent)
          w (:w @tu/agent)
          lens [74 64]
          rows (vec (for [r [0 1]]
                      (let [L (lens r)
                            k (count (filter pos? (nth (layers :marker-mask) r)))]
                        {:ids (vec (take L (map int (nth (layers :ids) r))))
                         :att (vec (repeat L 1))
                         :markers (vec (take k (nth (layers :marker-pos) r)))
                         :marker-mask (vec (repeat k 1))
                         :qtype (nth (layers :qtype) r)})))
          close (fn [a b tol] (every? true? (map #(<= (Math/abs (- (double %1) (double %2)))
                                                      (* tol (max 1.0 (Math/abs (double %2)))))
                                                 a b)))
          cpu (m/forward-batch w cfg rows)
          gpu ((:forward @f32) (:w @f32) cfg rows)
          gold-logits (t/load-file (str golden-dir "/layers/logits.f32") [2 4])]
      (is (= 2 (count gpu)))
      (doseq [r [0 1]]
        (let [[lg act] (nth gpu r)
              [lg-c act-c] (nth cpu r)
              k (count (:markers (nth rows r)))]
          (is (= k (count lg)) (str "row " r " answers one logit per marker"))
          (is (close lg lg-c 1e-4) (str "row " r " logits: mlx " lg " cpu " lg-c))
          (is (close act act-c 1e-4) (str "row " r " act logits: mlx " act " cpu " act-c))
          (is (close lg (map #(t/get (t/ptr gold-logits) (+ (* r 4) %)) (range k)) 1e-3)
              (str "row " r " logits vs torch's padded batch"))))
      (testing "a batch of one and a swapped batch answer the same rows"
        (let [[[lg-b _]] ((:forward @f32) (:w @f32) cfg [(rows 1)])
              swapped ((:forward @f32) (:w @f32) cfg [(rows 1) (rows 0)])]
          (is (close lg-b (first (nth gpu 1)) 1e-5))
          (is (close (first (second swapped)) (first (nth gpu 0)) 1e-5))))
      (testing "the full last head layer (:selected-head false) answers the same"
        (let [whole (mlx/load-agent data-dir {:name "english" :selected-head false})
              out ((:forward whole) (:w whole) cfg rows)]
          (is (false? (:selected-head (:w whole))))
          (is (true? (:selected-head (:w @f32))))
          (doseq [r [0 1]]
            (is (close (first (nth out r)) (first (nth gpu r)) 1e-5) (str "row " r))
            (is (close (second (nth out r)) (second (nth gpu r)) 1e-5) (str "row " r " act")))
          ((:close whole) whole)))
      (testing "f16: the same argmax, values within 1e-2"
        (let [half ((:forward @f16) (:w @f16) cfg rows)
              argmax (fn [xs] (first (apply max-key second (map-indexed vector xs))))]
          (doseq [r [0 1]]
            (is (= (argmax (first (nth cpu r))) (argmax (first (nth half r)))))
            (is (close (first (nth half r)) (first (nth cpu r)) 1e-2) (str "row " r " f16 " (first (nth half r)) " f32 " (first (nth cpu r))))))))))

(deftest answers-match-the-oracle-on-mlx
  (when (mlx/available?)
    (let [state (:readme-state @cases)
          questions (:readme-questions @cases)
          want (:system-one (tu/read-golden "readme"))]
      (testing "f32: the checkpoint's own answers to the fourth decimal, through the shared system-one"
        (let [out (ag/system-one @f32 state questions)]
          (is (= "english" (get out "model")))
          (tu/answers-match want (seq/json-str out))))
      (testing "f16: the same choices, the probabilities within 1e-2"
        (let [out (ag/system-one @f16 state questions)]
          (doseq [[qid a] (get out "answers")]
            (let [g (get-in (json/read-str want) ["answers" qid])]
              (is (= (get g "choice") (get a "choice")) qid)
              (when (get a "probabilities")
                (doseq [[opt p] (get a "probabilities")]
                  (is (<= (Math/abs (- (double p) (double (get-in g ["probabilities" opt])))) 1e-2) (str qid " " opt))))
              (when (contains? a "noul")
                (is (<= (Math/abs (- (double (get a "noul")) (double (get g "noul")))) 1e-2) qid)))))))))

(deftest the-router-loads-an-mlx-agent-on-request
  (when (mlx/available?)
    (let [r (router/make-router {:models {"english" data-dir} :max-loaded 1
                                 :checkpoints {"english" {:backend :mlx :dtype :f16}}})
          a (router/load-model r "english")]
      (is (= :mlx (:backend a)))
      (is (= :f16 (:dtype a)))
      (is (= "english" (:name a)))
      (router/unload r)
      (is (= [] (router/loaded r))))))
