(ns lev.snake-policy
  "lev's snake brain, the port of laya-mlx's snake/policy.py: the compact
  prompt and typed question set over the deterministic game, then guarded
  execution — the model proposes (argmax over the choice probabilities),
  the safety shield disposes (restricts to the planner's safe moves).
  The Decision map carries what the UI and the replay log need."
  (:require [lev.agent :as ag]
            [lev.snake :as s]
            [lev.sequence :as seq]))

(defn state-line
  "The compact state string (LayaPolicy.decide, prompt \"compact\"): two
  facts about the board, no length or open-cell counts."
  [safe? food-reachable?]
  (format "Safe route: %s. Food reachable through empty cells: %s."
          (if safe? "yes" "no") (if food-reachable? "yes" "no")))

(defn planner-best
  "The planner's preferred direction among the safe moves (max advance
  along the cycle); nil when nothing is safe."
  [moves]
  (when-let [safe (seq (filter :safe moves))]
    (:direction (apply max-key :advance safe))))

(defn compact-criteria
  "One line per direction, keyed for the choice question (the compact
  variant of the criteria descriptions)."
  [moves preferred]
  (seq/ordered-map
   (map (fn [{:keys [direction legal safe eats reason]}]
          [direction
           (cond
             (not legal) (str "Blocked. " reason ".")
             (not safe) "Unsafe. Traps the snake."
             eats "Safe. Eat food now. Best."
             (= direction preferred) "Safe. Best route to food."
             :else "Safe. Slower route.")])
        moves)))

(defn questions
  "The typed question set for one decision (the compact variant): a
  choice over the four directions with safety-annotated criteria, and
  two noul questions (risk = a safe route exists, food = food reachable)."
  [game]
  (let [moves (s/moves game)
        safe? (boolean (some :safe moves))
        [food-reach _] (s/food-reachability game)]
    (seq/ordered-map
     [["move" {:type "choice"
               :instructions "Choose the best safe move toward food."
               :criteria (compact-criteria moves (planner-best moves))}]
      ["risk" {:type "noul" :instructions "Is a safe route available?"}]
      ["food" {:type "noul" :instructions "Is food reachable through empty cells?"}]])))

(defn- argmax-direction [probs]
  (reduce (fn [best k] (if (> (get probs k 0.0) (get probs best 0.0)) k best))
          (keys probs)))

(defn- probability-error [p]
  (or (not (number? p))
      (let [d (double p)]
        (or (Double/isNaN d) (Double/isInfinite d) (neg? d) (> d 1.0)))))

(defn- valid-answer? [answers]
  (let [move (get answers "move")
        probs (get move "probabilities")
        risk (get-in answers ["risk" "noul"])
        food (get-in answers ["food" "noul"])]
    (and (map? probs)
         (= 4 (count probs))
         (every? (fn [v] (not (probability-error v))) (vals probs))
         (not (probability-error risk))
         (not (probability-error food)))))

(defn decide
  "One decision ([agent game] [agent game {:guarded true :prompt :compact}]):
  ask system-one the compact questions, take the move question's argmax as
  the proposal, and (guarded, the default) restrict execution to the
  planner's safe directions — the shield that keeps a wrong model alive.
  Returns the Decision: probabilities, proposed, executed, intervened,
  safe-directions, dead-end-risk (1 - risk noul), food-reachable,
  inference-ms, tokens, planner-best."
  ([agent game] (decide agent game {:guarded true}))
  ([agent game {:keys [guarded] :or {guarded true}}]
   (let [start (System/nanoTime)
         moves (s/moves game)
         safe (filter :safe moves)
         [food-reach] (s/food-reachability game)
         qs (questions game)
         t0 (System/nanoTime)
         output (ag/system-one agent (state-line (seq safe) food-reach) qs)
         inference-ms (/ (- (System/nanoTime) t0) 1e6)
         answers (get output "answers")
         probs (get-in answers ["move" "probabilities"])
         risk (get-in answers ["risk" "noul"])
         food (get-in answers ["food" "noul"])]
     (when-not (valid-answer? answers)
       (throw (ex-info "model returned an invalid probability; no move executed"
                       {:answers answers})))
     (let [proposed (argmax-direction probs)
           safe-dirs (mapv :direction safe)
           executed (if (and guarded (not (some #{proposed} safe-dirs)))
                      (argmax-direction (select-keys probs safe-dirs))
                      proposed)]
       {:probabilities probs
        :proposed proposed
        :executed executed
        :intervened (not= proposed executed)
        :safe-directions safe-dirs
        :dead-end-risk (- 1.0 risk)
        :food-reachable food
        :inference-ms inference-ms
        :decision-ms (/ (- (System/nanoTime) start) 1e6)
        :input-tokens (get-in output ["usage" "input_tokens"])
        :planner-best (planner-best moves)})))
)
