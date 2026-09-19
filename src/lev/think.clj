(ns lev.think
  "The thinker: system-one over a generative model (lev.llm), for the
  cases an encoder gets wrong. The same state and typed questions in, the
  same answer shapes out; what differs is the time (seconds a question
  with thinking, ~100 ms without on a GPU) and how the probabilities
  arise: per question the model reads a chat prompt with the state, the
  instructions and every option, thinks (or not), is handed the answer
  prefix, and each option is then scored by the log probability of its
  tokens — a softmax over those is the answer's distribution. On the
  authored144 set (bench/) MiniCPM5-2B answers 95% with thinking and
  74% without, against 61-76% for the encoders.

  A thinker is data: {:kind :thinker :name :cfg :decide :count-tokens},
  with `decide` (prompt options opts -> {:logp :thought :tokens}) the
  model behind it, so the engine is tested with a fake one and the real
  one is lev.llm/decide. `thinker` builds either; lev.router loads the
  configured ones by name."
  (:require [clojure.string :as str]
            [lev.agent :as ag]
            [lev.llm :as llm]
            [lev.sequence :as seq]))

(def defaults
  {:thinking true
   :max-think-tokens 2048
   :n-ctx 4096
   :n-gpu-layers -1
   :threads 0
   ;; greedy: on authored144 a sampled thought (temperature 1.0, MiniCPM's
   ;; general setting) lands at 91%, the greedy one at 95%, and greedy is
   ;; reproducible
   :temperature 0.0
   :top-p 0.95
   :min-p 0.0
   :seed 42
   :system "You are a careful decision model. Read the state, then answer the question by choosing exactly one of the options."})

(defn thinker
  "A thinker agent from its config ({:name :model (a GGUF path) :thinking
  :max-think-tokens :n-ctx :n-gpu-layers :threads :temperature :top-p
  :min-p :seed :system}); with `backing` ({:decide :count-tokens}) the
  model is whatever those fns are (tests), else the GGUF is loaded through
  lev.llm."
  ([cfg] (thinker cfg nil))
  ([cfg backing]
   (let [cfg (merge defaults cfg)
         backing (or backing
                     (let [m (llm/load (:model cfg) (select-keys cfg [:n-ctx :n-gpu-layers :threads :n-seq-max]))]
                       {:llm m
                        :decide (fn [prompt options opts] (llm/decide m prompt options opts))
                        :count-tokens (fn [text] (llm/count-tokens m text))}))]
     (merge {:kind :thinker :name (or (:name cfg) "thinker") :cfg cfg} backing))))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn options-for
  "[[id description] ...] the model is shown and the ids it is scored on:
  a choice's options, a score's level indices with their legend, a noul's
  true/false with its criteria."
  [q]
  (case (:t q)
    "choice" (mapv (fn [[c d]] [(key-str c) d]) (:crit q))
    "score" (vec (map-indexed (fn [i d] [(str i) d]) (:crit q)))
    (let [crit (:crit q)]
      [["true" (or (get crit "true") (get crit :true) "the statement holds")]
       ["false" (or (get crit "false") (get crit :false) "the statement does not hold")]])))

(defn- question-line [q]
  (case (:t q)
    "choice" "Choose the one option that fits best."
    "score" "Choose the level (a number) that fits best."
    "Decide whether the statement is true or false."))

(defn messages
  "The chat for one question: the state as the model reads it (serialized
  like the encoders'), the instructions, the options with descriptions."
  [cfg state q]
  [{:role "system" :content (:system cfg)}
   {:role "user"
    :content (str "State:\n" (seq/serialize-state state)
                  "\n\nQuestion: " (:ins q)
                  "\n" (question-line q)
                  "\n\nOptions:\n"
                  (str/join "\n" (map (fn [[id d]] (if (str/blank? (str d)) (str "- " id) (str "- " id ": " d)))
                                      (options-for q)))
                  "\n\nReply with ANSWER: <option id>.")}])

(defn- ask
  "One question through the model: its prompt, options and the scoring."
  [{:keys [cfg decide count-tokens]} state q thinking?]
  (let [prompt (llm/chat-prompt nil (messages cfg state q) {:thinking thinking?})
        ids (mapv first (options-for q))
        think-max (if thinking? (:max-think-tokens cfg) 0)
        {:keys [logp thought tokens]} (decide prompt ids
                                              {:think-max think-max
                                               :temperature (:temperature cfg) :top-p (:top-p cfg)
                                               :min-p (:min-p cfg) :seed (:seed cfg)})
        p (llm/softmax logp)]
    {:p (if (= "noul" (:t q)) [(nth p 1) (nth p 0)] p)   ; answer order: [false true] for a noul
     :k (count ids)
     :thought thought
     :thought-tokens tokens
     :prompt-tokens (count-tokens prompt)}))

(defmethod ag/system-one* :thinker
  [{:keys [cfg name] :as t} state questions {:keys [constraints on-infeasible thinking thought]}]
  (let [thinking? (if (some? thinking) (boolean thinking) (boolean (:thinking cfg)))
        prepared (mapv (fn [[qid qdef]]
                         (let [qdef (ag/validate-question qid qdef)]
                           {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
                       questions)
        cs (ag/prepare-constraints (map (fn [{:keys [qid qdef]}] [qid qdef]) prepared) constraints)
        asked (mapv (fn [{:keys [q]}] (ask t state q thinking?)) prepared)
        solution (when cs
                   (ag/decide-constraints cs (map (fn [{:keys [qid q]} {:keys [p]}] [qid q p]) prepared asked)
                                          on-infeasible))
        answers (seq/ordered-map
                 (map (fn [{:keys [qid q]} {:keys [p k] :as a}]
                        [qid (ag/typed-answer q p k (when solution (ag/decided-label cs solution qid))
                                              (when thought [["thought" (:thought a)]]))])
                      prepared asked))
        out-tokens (reduce + (map :thought-tokens asked))]
    (seq/ordered-map
     (concat [["model" name]
              ["answers" answers]
              ["usage" (array-map "input_tokens" (reduce + (map :prompt-tokens asked))
                                  "output_tokens" out-tokens)]
              ["thinking" (array-map "enabled" thinking?
                                     "tokens" out-tokens
                                     "max_tokens" (if thinking? (:max-think-tokens cfg) 0))]]
             (when solution [["constraints" (ag/constraints-report cs solution)]])))))

(defn system-one
  "state + {qid -> qdef} -> the answer map, from a thinker. opts: those of
  lev.agent/system-one plus :thinking (override the thinker's default)
  and :thought (true: each answer carries the model's thought text)."
  [thinker state questions opts]
  (ag/system-one* thinker state questions opts))
