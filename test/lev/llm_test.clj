(ns lev.llm-test
  "lev.llm: llama.cpp behind native/lev_llm.c. The real model tests need a
  GGUF (LEV_TEST_GGUF, else ~/src/models/MiniCPM5-2B-Q8_0.gguf) and the
  llm native (jolt llama); without either they are skipped, and only the
  availability contract is checked."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [lev.llm :as llm]))

(def gguf
  (let [p (or (System/getenv "LEV_TEST_GGUF")
              (str (System/getProperty "user.home") "/src/models/MiniCPM5-2B-Q8_0.gguf"))]
    (when (.exists (io/file p)) p)))

(def model
  "Loaded once for the suite: ~2.7 GB read, a second or two."
  (delay (llm/load gguf {:n-ctx 2048 :n-seq-max 8})))

(deftest availability-is-a-question-not-a-crash
  (is (boolean? (llm/available?)))
  (when-not (llm/available?)
    (is (thrown-with-msg? Exception #"jolt llama" (llm/load "x.gguf" {})))))

(deftest a-missing-file-is-an-error-with-the-path
  (when (llm/available?)
    (let [e (try (llm/load "target/nope.gguf" {}) nil (catch Exception e e))]
      (is (some? e))
      (is (= :model-unavailable (:type (ex-data e))))
      (is (str/includes? (ex-message e) "nope.gguf")))))

(deftest generates-text
  (when (and (llm/available?) gguf)
    (let [m @model]
      (is (llm/ok? m))
      (is (>= (llm/n-ctx m) 2048))
      (is (str/starts-with? (llm/version) "llama.cpp"))
      (testing "greedy completion of a chat prompt, thinking off"
        (let [prompt (llm/chat-prompt m [{:role "user" :content "Reply with the single word: pong"}] {:thinking false})
              {:keys [text tokens]} (llm/generate m prompt {:max-tokens 8 :temperature 0.0})]
          (is (pos? tokens))
          (is (str/includes? (str/lower-case text) "pong") (pr-str text))))
      (testing "token counting"
        (is (< 3 (llm/count-tokens m "one two three four five") 12))))))

(deftest decides-among-options
  (when (and (llm/available?) gguf)
    (let [m @model
          state "The optician ordered replacement lenses. The workshop confirms they have not yet been fitted to the customer's glasses."
          question "Assess the claim: the replacement lenses have been fitted."
          options ["supported" "insufficient" "contradicted"]
          ask (fn [thinking?]
                (llm/decide m (llm/chat-prompt m [{:role "user"
                                                  :content (str "State:\n" state "\n\nQuestion: " question
                                                                "\n\nOptions: " (str/join ", " options)
                                                                "\n\nReply with ANSWER: <option>.")}]
                                              {:thinking thinking?})
                            options {:think-max (if thinking? 512 0) :temperature 0.0}))]
      (testing "without thinking: log probabilities for every option, no thought (the answer itself is the model's, and direct answers are right ~74% of the time on this set)"
        (let [{:keys [logp thought tokens]} (ask false)]
          (is (= 3 (count logp)))
          (is (every? #(and (number? %) (neg? %)) logp))
          (is (= 0 tokens))
          (is (= "" thought))
          (is (< (Math/abs (- 1.0 (reduce + (llm/softmax logp)))) 1e-9))))
      (testing "with thinking: a thought that ends with the closing tag, then the same scoring"
        (let [{:keys [logp thought tokens]} (ask true)]
          (is (pos? tokens))
          (is (str/ends-with? (str/trimr thought) "</think>") (subs thought (max 0 (- (count thought) 80))))
          (is (= "contradicted" (nth options (llm/argmax logp)))))))))
