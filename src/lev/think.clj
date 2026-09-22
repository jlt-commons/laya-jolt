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
  75% without, against 61-76% for the encoders.

  Thinking off, the call's questions go to lev.llm/jev together (Jev
  mode, :jev in the config): the same prompts, cut at the state, so the
  chat before it is decoded once and kept, the state once a call, and
  every question's tail and options in one batch. Four questions over a
  short email: 207 ms against 539 ms one prompt at a time.

  A thinker is data: {:kind :thinker :name :cfg :decide :jev :escape
  :count-tokens}, with `decide` (prompt options opts -> {:logp :thought
  :tokens}) and `jev` (lev.llm/jev's request -> its answer) the model
  behind it and `escape` its lev.llm/escape, so the engine is tested with
  a fake one and the real one is lev.llm. `thinker` builds either; lev.router loads the
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
   ;; thinking off, every question of a call is answered in one pass by
   ;; the vendored llama.cpp fork's decision engine (lev.llm/jev) instead
   ;; of one prompt per question; false keeps the per-question path
   :jev true
   ;; the option ids start a token of their own after the answer prefix
   ;; (the per-question path's cut), not merged with its trailing space:
   ;; on authored144 the merged cut answers 68.1% (+15 -23 against the
   ;; per-question path), this one 75.0% (+3 -1), bench/README.md
   :split-boundary true
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
                        :jev (fn [req] (llm/jev m req))
                        :escape (fn [text] (llm/escape m text))
                        :count-tokens (fn [text] (llm/count-tokens m text))
                        ;; the router calls this when it evicts or unloads the thinker
                        :close (fn [_] (llm/free! m))}))]
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
  like the encoders'), the instructions, the options with descriptions.
  `esc` (identity by default) makes caller text safe to tokenize with the
  chat's special tokens parsed (lev.llm/escape); `state-text`, when given,
  stands in for the serialized state."
  ([cfg state q] (messages cfg state q identity nil))
  ([cfg state q esc state-text]
   [{:role "system" :content (:system cfg)}
    {:role "user"
     :content (str "State:\n" (or state-text (esc (seq/serialize-state state)))
                   "\n\nQuestion: " (esc (str (:ins q)))
                   "\n" (question-line q)
                   "\n\nOptions:\n"
                   (str/join "\n" (map (fn [[id d]] (let [id (esc id)]
                                                      (if (str/blank? (str d)) (str "- " id) (str "- " id ": " (esc (str d))))))
                                       (options-for q)))
                   "\n\nReply with ANSWER: <option id>.")}]))

(defn- escaper [t] (or (:escape t) identity))

(defn- as-answer
  "The typed answer's distribution from one over the option ids."
  [q p]
  (if (= "noul" (:t q)) [(nth p 1) (nth p 0)] p))   ; answer order: [false true] for a noul

(defn- ask
  "One question through the model: its prompt, options and the scoring."
  [{:keys [cfg decide count-tokens] :as t} state q thinking?]
  (let [prompt (llm/chat-prompt nil (messages cfg state q (escaper t) nil) {:thinking thinking?})
        ids (mapv first (options-for q))
        think-max (if thinking? (:max-think-tokens cfg) 0)
        {:keys [logp thought tokens]} (decide prompt ids
                                              {:think-max think-max
                                               :temperature (:temperature cfg) :top-p (:top-p cfg)
                                               :min-p (:min-p cfg) :seed (:seed cfg)})]
    {:p (as-answer q (llm/softmax logp))
     :k (count ids)
     :thought thought
     :thought-tokens tokens
     :prompt-tokens (count-tokens prompt)}))

(def ^:private state-mark "\u001f<<lev-state>>\u001f")

(defn- ask-all
  "Every question in one Jev-mode call: each question's direct prompt
  (thinking off) cut at the state, so the chat before it is the shared
  text, the state the context, and the rest of the prompt through the
  answer prefix the field whose values are the option ids closed by the
  turn's end. Same prompts as `ask`, one pass."
  [{:keys [cfg jev] :as t} state qs]
  (let [esc (escaper t)
        {:keys [answer-prefix answer-end]} llm/defaults
        parts (mapv (fn [q]
                      (let [full (llm/chat-prompt nil (messages cfg state q esc state-mark) {:thinking false})
                            i (str/index-of full state-mark)]
                        [(subs full 0 i) (str (subs full (+ i (count state-mark))) answer-prefix)]))
                    qs)
        shared (ffirst parts)
        context (esc (seq/serialize-state state))
        {:keys [probs context-tokens shared-tokens rows]}
        (jev {:shared shared
              :contexts [context]
              :split-boundary? (:split-boundary cfg)
              :fields (mapv (fn [q [_ suffix]]
                              {:suffix suffix :values (mapv #(str (esc (first %)) answer-end) (options-for q))})
                            qs parts)})
        input-tokens (+ shared-tokens (first context-tokens) rows)]
    (mapv (fn [q p i]
            {:p (as-answer q p)
             :k (count p)
             :thought ""
             :thought-tokens 0
             ;; the call's decoded tokens, counted once, on the first question
             :prompt-tokens (if (zero? i) input-tokens 0)})
          qs (first probs) (range))))

(defmethod ag/system-one* :thinker
  [{:keys [cfg name] :as t} state questions {:keys [constraints on-infeasible thinking thought]}]
  (let [thinking? (if (some? thinking) (boolean thinking) (boolean (:thinking cfg)))
        prepared (mapv (fn [[qid qdef]]
                         (let [qdef (ag/validate-question qid qdef)]
                           {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
                       questions)
        cs (ag/prepare-constraints (map (fn [{:keys [qid qdef]}] [qid qdef]) prepared) constraints)
        asked (if (and (not thinking?) (:jev cfg) (:jev t) (seq prepared))
                (ask-all t state (mapv :q prepared))
                (mapv (fn [{:keys [q]}] (ask t state q thinking?)) prepared))
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
