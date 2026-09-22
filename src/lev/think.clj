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
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.llm :as llm]
            [lev.sequence :as seq]))

(def defaults
  {:thinking true
   ;; false: the model has no thinking mode (Qwen2.5-Instruct, say): its
   ;; prompts carry no <think> tags, the answer prefix follows the
   ;; assistant turn's opening directly, and thinking stays off whatever
   ;; a request asks
   :thinks true
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
   ;; "lev" (the state, the question, the options by id, ANSWER: <id>) or
   ;; "semif" (SemIf's direct prompt: a JSON evidence / criterion /
   ;; lettered options payload, the letter scored); per model, measured
   ;; in bench/README.md
   :prompt "lev"
   ;; "question" (each question's whole prompt after the state) or
   ;; "catalog" (every question before the state, a short field per
   ;; question after it: faster in a batch, a different prompt)
   :layout "question"
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

(def semif-system
  "SemIf's direct-readout instruction (github.com/TheoLeeCJ/SemIf,
  core.DIRECT_SYSTEM)."
  "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.")

(def ^:private letters (mapv str "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

(defn- semif? [cfg] (= "semif" (some-> (:prompt cfg) name)))

(defn- jstr
  "A JSON string as Python's json.dumps(ensure_ascii=False) writes it."
  [s]
  (json/write-str (str s) :escape-unicode false :escape-slash false))

(defn- state-text
  "The serialized state as the prompt carries it: escaped, and for the
  semif prompt a JSON string."
  [cfg esc state]
  (let [t (esc (seq/serialize-state state))]
    (if (semif? cfg) (jstr t) t)))

(defn messages
  "The chat for one question: the state as the model reads it (serialized
  like the encoders'), the instructions, the options with descriptions.
  `esc` (identity by default) makes caller text safe to tokenize with the
  chat's special tokens parsed (lev.llm/escape); `state-text`, when given,
  stands in for the state as the prompt carries it. With :prompt
  \"semif\" it is SemIf's JSON payload with lettered options."
  ([cfg state q] (messages cfg state q identity nil))
  ([cfg state q esc stext]
   (let [stext (or stext (state-text cfg esc state))]
     (if (semif? cfg)
       [{:role "system" :content semif-system}
        {:role "user"
         :content (str "{\"evidence\": " stext
                       ", \"criterion\": " (jstr (esc (str (:ins q))))
                       ", \"options\": ["
                       (str/join ", " (map-indexed (fn [i [id d]]
                                                     (str "{\"letter\": \"" (letters i) "\", \"description\": "
                                                          (jstr (esc (str (if (str/blank? (str d)) id d)))) "}"))
                                                   (options-for q)))
                       "]}")}]
       [{:role "system" :content (:system cfg)}
        {:role "user"
         :content (str "State:\n" stext
                       "\n\nQuestion: " (esc (str (:ins q)))
                       "\n" (question-line q)
                       "\n\nOptions:\n"
                       (str/join "\n" (map (fn [[id d]] (let [id (esc id)]
                                                          (if (str/blank? (str d)) (str "- " id) (str "- " id ": " (esc (str d))))))
                                           (options-for q)))
                       "\n\nReply with ANSWER: <option id>.")}]))))

(defn- answer-ids
  "What the model is scored on for a question: the option ids, or for the
  semif prompt their letters (in the same order)."
  [cfg esc q]
  (if (semif? cfg)
    (subvec letters 0 (count (options-for q)))
    (mapv (comp esc first) (options-for q))))

(defn- escaper [t] (or (:escape t) identity))

(defn- thinks?
  "Does the model have a thinking mode? (:thinks, true unless false.)"
  [cfg]
  (not (false? (:thinks cfg))))

(defn- chat-thinking
  "chat-prompt's :thinking for a call: the model's switch when it has one,
  no tags at all when it does not."
  [cfg thinking?]
  (when (thinks? cfg) (boolean thinking?)))

(defn- answer-prefix
  "What is forced before the options: after the closed thought's tag,
  \"\\n\\nANSWER: \" (the template's shape); straight after the assistant
  turn's opening for a model without thoughts."
  [cfg]
  (cond
    (semif? cfg) (if (thinks? cfg) "\n\n" "")
    (thinks? cfg) (:answer-prefix llm/defaults)
    :else "ANSWER: "))

(defn- as-answer
  "The typed answer's distribution from one over the option ids."
  [q p]
  (if (= "noul" (:t q)) [(nth p 1) (nth p 0)] p))   ; answer order: [false true] for a noul

(defn- ask
  "One question through the model: its prompt, options and the scoring."
  [{:keys [cfg decide count-tokens] :as t} state q thinking?]
  (let [prompt (llm/chat-prompt nil (messages cfg state q (escaper t) nil) {:thinking (chat-thinking cfg thinking?)})
        ids (answer-ids cfg (escaper t) q)
        think-max (if thinking? (:max-think-tokens cfg) 0)
        {:keys [logp thought tokens]} (decide prompt ids
                                              {:think-max think-max
                                               :answer-prefix (answer-prefix cfg)
                                               :temperature (:temperature cfg) :top-p (:top-p cfg)
                                               :min-p (:min-p cfg) :seed (:seed cfg)})]
    {:p (as-answer q (llm/softmax logp))
     :k (count ids)
     :thought thought
     :thought-tokens tokens
     :prompt-tokens (count-tokens prompt)}))

(def ^:private state-mark "\u001f<<lev-state>>\u001f")

(defn- question-parts
  "The question layout: each question's direct prompt cut at the state,
  the chat before it shared, the rest of it through the answer prefix one
  field each."
  [cfg qs esc]
  (let [prefix (answer-prefix cfg)
        parts (mapv (fn [q]
                      (let [full (llm/chat-prompt nil (messages cfg nil q esc state-mark) {:thinking (chat-thinking cfg false)})
                            i (str/index-of full state-mark)]
                        [(subs full 0 i) (str (subs full (+ i (count state-mark))) prefix)]))
                    qs)]
    {:shared (ffirst parts) :closing "" :suffixes (mapv second parts)}))

(defn- catalog-parts
  "The catalog layout (the fork's own shape): every question in the shared
  text before the state, the turn's end after the state, and one short
  field per question, `ANSWER <number>: `. Fewer tokens a branch, so
  a batch of states costs little more than its states; a different
  prompt, so its own numbers (bench/README.md)."
  [cfg qs esc]
  (let [catalog (str/join "\n\n"
                          (map-indexed
                           (fn [i q]
                             (str "Question " (inc i) ": " (esc (str (:ins q))) "\n" (question-line q) "\nOptions:\n"
                                  (str/join "\n" (map (fn [[id d]] (let [id (esc id)]
                                                                     (if (str/blank? (str d)) (str "- " id) (str "- " id ": " (esc (str d))))))
                                                      (options-for q)))))
                           qs))
        full (llm/chat-prompt nil [{:role "system" :content (:system cfg)}
                                   {:role "user" :content (str catalog "\n\nState:\n" state-mark
                                                               "\n\nAnswer every question with ANSWER <question number>: <option id>.")}]
                              {:thinking (chat-thinking cfg false)})
        i (str/index-of full state-mark)
        prefix (answer-prefix cfg)]
    {:shared (subs full 0 i)
     :closing (subs full (+ i (count state-mark)))
     :suffixes (mapv #(str (str/replace prefix "ANSWER: " (str "ANSWER " (inc %) ": "))) (range (count qs)))}))

(defn- ask-all
  "Every question for every state in one Jev-mode call: each question's
  direct prompt (thinking off) cut at the state, so the chat before it is
  the shared text, each state a context, and the rest of the prompt
  through the answer prefix the field whose values are the option ids
  closed by the turn's end. Same prompts as `ask`, one pass. Per state,
  the asked questions."
  [{:keys [cfg jev] :as t} states qs]
  (let [catalog? (= "catalog" (some-> (:layout cfg) name))
        ;; the catalog has a prompt of its own, which scores the ids
        cfg (if catalog? (assoc cfg :prompt "lev") cfg)
        esc (escaper t)
        {:keys [answer-end]} llm/defaults
        {:keys [shared closing suffixes]} (if catalog?
                                            (catalog-parts cfg qs esc)
                                            (question-parts cfg qs esc))
        {:keys [probs context-tokens shared-tokens rows]}
        (jev {:shared shared
              :contexts (mapv #(str (state-text cfg esc %) closing) states)
              :split-boundary? (:split-boundary cfg)
              :fields (mapv (fn [q suffix]
                              {:suffix suffix :values (mapv #(str % answer-end) (answer-ids cfg esc q))})
                            qs suffixes)})
        ;; the rows are the same for every state
        per-state-rows (/ rows (count states))]
    (mapv (fn [ps ctoks si]
            (mapv (fn [q p qi]
                    {:p (as-answer q p)
                     :k (count p)
                     :thought ""
                     :thought-tokens 0
                     ;; a state's decoded tokens, on its first question; the
                     ;; shared text's on the first state's
                     :prompt-tokens (if (zero? qi) (+ ctoks per-state-rows (if (zero? si) shared-tokens 0)) 0)})
                  qs ps (range)))
          probs context-tokens (range))))

(defn- prepare
  "The call's validated questions and constraints."
  [questions constraints]
  (let [prepared (mapv (fn [[qid qdef]]
                         (let [qdef (ag/validate-question qid qdef)]
                           {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
                       questions)]
    {:prepared prepared
     :cs (ag/prepare-constraints (map (fn [{:keys [qid qdef]}] [qid qdef]) prepared) constraints)}))

(defn- answer-map
  "The Jev answer map for one state's asked questions."
  [{:keys [cfg name]} {:keys [prepared cs]} asked thinking? {:keys [on-infeasible thought]}]
  (let [solution (when cs
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

(defn- call-thinking
  "Does this call think? The request's :thinking over the thinker's
  default, and never for a model without a thinking mode."
  [cfg thinking]
  (and (thinks? cfg) (if (some? thinking) (boolean thinking) (boolean (:thinking cfg)))))

(defmethod ag/system-one-batch* :thinker
  [{:keys [cfg] :as t} states questions {:keys [constraints thinking] :as opts}]
  (let [thinking? (call-thinking cfg thinking)
        {:keys [prepared] :as p} (prepare questions constraints)
        qs (mapv :q prepared)
        asked (if (and (not thinking?) (:jev cfg) (:jev t) (seq prepared) (seq states))
                (ask-all t states qs)
                (mapv (fn [state] (mapv #(ask t state % thinking?) qs)) states))]
    (mapv #(answer-map t p % thinking? opts) asked)))

(defmethod ag/system-one* :thinker
  [t state questions opts]
  (first (ag/system-one-batch* t [state] questions opts)))

(defn system-one
  "state + {qid -> qdef} -> the answer map, from a thinker. opts: those of
  lev.agent/system-one plus :thinking (override the thinker's default)
  and :thought (true: each answer carries the model's thought text)."
  [thinker state questions opts]
  (ag/system-one* thinker state questions opts))
