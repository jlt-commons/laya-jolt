(ns lev.snake-policy-test
  "The lev.snake-policy layer over the game: compact prompt construction,
  the choice/noul question set, guarded execution against the safe moves.
  Logic tests run against a :mock agent registered on system-one*; one
  smoke test exercises the real english checkpoint (test-util's delay)."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.snake :as s]
            [lev.snake-policy :as p]
            [lev.test-util :as tu]))

(defn- install-mock!
  "Register an agent kind whose answers are fixed: move probabilities,
  risk and food noul values. Captures the state it was asked about."
  [probs risk food]
  (let [seen (atom nil)]
    (defmethod ag/system-one* :mock [_agent state questions _opts]
      (reset! seen state)
      (is (string? state))
      {"model" "mock"
       "answers" {"move" {"type" "choice" "choice" "UP" "probabilities" probs
                          "confidence" 0.5}
                  "risk" {"type" "noul" "noul" risk "confidence" 0.5}
                  "food" {"type" "noul" "noul" food "confidence" 0.5}}
       "usage" {"input_tokens" 10 "output_tokens" 0}})
    seen))

(defn- mock-agent [] {:kind :mock :name "mock"})

(deftest state-line
  (is (= "Safe route: yes. Food reachable through empty cells: yes."
         (p/state-line true true)))
  (is (= "Safe route: no. Food reachable through empty cells: no."
         (p/state-line false false)))
  (is (= "Safe route: yes. Food reachable through empty cells: no."
         (p/state-line true false))))

(deftest compact-criteria
  (let [g (-> (s/new-game) (assoc :body [[3 3] [3 4] [3 5]]) (assoc :food [2 3]))
        moves (s/moves g)
        preferred (p/planner-best moves)
        criteria (p/compact-criteria moves preferred)]
    (is (= ["UP" "DOWN" "LEFT" "RIGHT"] (vec (keys criteria))))
    (is (= "Safe. Eat food now. Best." (get criteria "LEFT")))
    (is (= "Blocked. reverse." (get criteria "DOWN")))
    (is (= "Unsafe. Traps the snake." (get criteria "UP")))
    (is (every? string? (vals criteria)))))

(deftest questions-shape
  (let [qs (p/questions (-> (s/new-game) (assoc :body [[3 3] [3 4] [3 5]])
                            (assoc :food [2 3])))]
    (is (= ["move" "risk" "food"] (vec (keys qs))))
    (is (= "choice" (:type (get qs "move"))))
    (is (= "noul" (:type (get qs "risk"))))
    (is (= "noul" (:type (get qs "food"))))
    (is (string? (:instructions (get qs "move"))))
    (is (= 4 (count (:criteria (get qs "move")))))))

(deftest decide-guard-executes-a-safe-move
  ;; fresh board: UP legal but unsafe, DOWN/RIGHT safe, LEFT reverse
  (let [seen (install-mock! {"UP" 0.6 "DOWN" 0.3 "LEFT" 0.05 "RIGHT" 0.05} 0.9 0.8)
        d (p/decide (mock-agent) (s/new-game))]
    (is (string? @seen) "the state line reached the agent")
    (is (= "UP" (:proposed d)) "argmax of the probabilities")
    (is (= "DOWN" (:executed d)) "the best safe direction instead")
    (is (true? (:intervened d)))
    (is (every? #(and (>= % 0) (<= % 1)) (vals (:probabilities d))))
    (is (< 0.0999 (:dead-end-risk d) 0.1001) "1 - risk noul")
    (is (= 0.8 (:food-reachable d)))))

(deftest decide-unguarded-trusts-the-model
  (install-mock! {"UP" 0.6 "DOWN" 0.3 "LEFT" 0.05 "RIGHT" 0.05} 0.9 0.8)
  (let [d (p/decide (mock-agent) (s/new-game) {:guarded false})]
    (is (= "UP" (:proposed d)))
    (is (= "UP" (:executed d)))
    (is (false? (:intervened d)))))

(deftest decide-validates-probabilities
  (install-mock! {"UP" 1.5 "DOWN" 0.6 "LEFT" -0.05 "RIGHT" 0.05} 0.9 0.8)
  (is (thrown? Exception (p/decide (mock-agent) (s/new-game)))))

(deftest decide-unguarded-unknown-direction-still-throws
  ;; probabilities that are not finite are rejected before any move
  (install-mock! {"UP" 0.6 "DOWN" 0.3 "LEFT" 0.05 "RIGHT" Double/NaN} 0.9 0.8)
  (is (thrown? Exception (p/decide (mock-agent) (s/new-game)))))

(deftest decide-against-the-real-agent
  (let [d (p/decide @tu/agent (s/new-game))]
    (is (= #{"UP" "DOWN" "LEFT" "RIGHT"} (set (keys (:probabilities d)))))
    (is (contains? (set s/directions) (:executed d)))
    (is (pos? (:inference-ms d)))))

(deftest decide-throws-when-nothing-is-safe
  (install-mock! {"UP" 0.3 "DOWN" 0.3 "LEFT" 0.2 "RIGHT" 0.2} 0.9 0.8)
  ;; head boxed by its own body ahead along the cycle: every legal move
  ;; crosses the tail, so nothing is safe
  (let [cycle (s/hamiltonian-cycle 6 6)
        body (vec (take 5 (drop 30 cycle)))
        g (-> (s/new-game 6 6 7 2) (assoc :food (nth cycle 3))
              (assoc :body body))]
    (is (empty? (filter :safe (s/moves g))))
    (is (thrown? Exception (p/decide (mock-agent) g)))))
