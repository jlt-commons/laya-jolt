(ns lev.snake-test
  "The deterministic snake rules, ported from laya-mlx's snake/game.py and
  pinned to its observable behavior: the hamiltonian cycle, legality with
  the tail vacating, the cycle-safety planner, flood-fill reachability."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.snake :as s]))

(deftest hamiltonian-cycle-shape
  (testing "visits every cell once, adjacent steps, closes the loop"
    (let [w 8 h 6
          cycle (s/hamiltonian-cycle w h)]
      (is (= (* w h) (count cycle)))
      (is (= (* w h) (count (set cycle))) "no cell repeats")
      (is (= [0 0] (first cycle)))
      ;; every step is a 4-neighbour move, closing edge included
      (is (every? (fn [[a b]]
                    (= 1 (+ (Math/abs (- (first a) (first b)))
                            (Math/abs (- (second a) (second b))))))
                  (map vector cycle (concat (rest cycle) [(first cycle)])))))))

(deftest hamiltonian-cycle-rejects-bad-boards
  (testing "min dimension 4, at least one even"
    (is (thrown? Exception (s/hamiltonian-cycle 3 6)))
    (is (thrown? Exception (s/hamiltonian-cycle 5 5)))
    (is (= 20 (count (s/hamiltonian-cycle 5 4))) "odd width, even height is fine")))

(deftest initial-game
  (let [g (s/new-game)]
    (testing "defaults"
      (is (= 24 (:width g)))
      (is (= 16 (:height g)))
      (is (= 7 (:seed g)))
      (is (= 6 (count (:body g))))
      (is (s/alive? g))
      (is (not (:won g)))
      (is (nil? (:death-reason g)))
      (is (= 0 (:score g)))
      (is (vector? (:food g)))
      (is (not (some #(= % (:food g)) (:body g)))))
    (testing "every body cell is on the board"
      (is (every? (fn [[x y]] (and (<= 0 x) (< x 24) (<= 0 y) (< y 16))) (:body g))))))

(deftest legality
  (let [g (s/new-game)]
    (testing "all four directions reported"
      (is (= ["UP" "DOWN" "LEFT" "RIGHT"] (mapv :direction (s/moves g)))))
    (testing "reasons partition legality (a legal-but-unsafe move carries a safety reason)"
      (is (every? (fn [{:keys [legal reason]}]
                    (if legal
                      (not (contains? #{"wall" "reverse" "body"} reason))
                      (contains? #{"wall" "reverse" "body"} reason)))
                  (s/moves g))))
    (testing "at least one legal move on the fresh board"
      (is (some :legal (s/moves g))))))

(deftest step-eats-grows-and-respawns-food
  (let [g (-> (s/new-game) (assoc :body [[3 3] [3 4] [3 5]]) (assoc :food [3 2]))
        eat (some #(when (:eats %) %) (s/moves g))]
    (is (some? eat) "food adjacent to the head: UP eats")
    (is (= "UP" (:direction eat)))
    (let [g2 (s/step g "UP")]
      (is (= [3 2] (s/head g2)))
      (is (= 1 (:score g2)))
      (is (= 4 (count (:body g2))))
      (is (not= [3 2] (:food g2)) "food respawns on an empty cell")
      (is (not (some #(= % (:food g2)) (:body g2)))))))

(deftest step-non-eating-moves-the-tail
  (let [g (-> (s/new-game) (assoc :body [[3 3] [3 4] [3 5]]) (assoc :food [0 0]))
        plain (some #(when (and (:legal %) (not (:eats %))) %) (s/moves g))]
    (is (some? plain))
    (let [g2 (s/step g (:direction plain))]
      (is (= (count (:body g)) (count (:body g2))))
      (is (= (butlast (:body g)) (rest (:body g2))) "old body minus tail = new body minus head"))))

(deftest step-into-wall-dies
  (let [g (-> (s/new-game) (assoc :body [[0 0] [0 1] [0 2]]) (assoc :food [5 5]))
        dead (s/step g "LEFT")]
    (is (not (s/alive? dead)))
    (is (= "wall" (:death-reason dead)))
    (is (thrown? Exception (s/step dead "RIGHT")) "cannot step a finished game")))

(deftest reverse-is-illegal
  (let [g (-> (s/new-game) (assoc :body [[3 3] [3 4] [3 5]]) (assoc :food [0 0]))]
    (is (= "reverse" (:reason (some #(when (= "DOWN" (:direction %)) %) (s/moves g)))))))

(deftest safety-planner-marks-unsafe-moves
  (let [g (s/new-game)]
    (testing "every legal move is either safe or carries a safety reason"
      (is (every? (fn [{:keys [legal safe reason]}]
                    (or (not legal)
                        safe
                        (contains? #{"would cross the tail"
                                     "would skip the food on the safe route"} reason)))
                  (s/moves g)))
      (is (some :safe (s/moves g))))))

(deftest food-reachability-bfs
  (let [g (s/new-game)
        [reach space] (s/food-reachability g)]
    (is (true? reach))
    (is (pos? space))
    (is (< space (* (:width g) (:height g))))))

(deftest deterministic-replay
  (testing "same seed, same greedy-safe walk"
    (let [run (fn [] (loop [g (s/new-game) n 0]
                      (if (or (not (s/alive? g)) (:won g) (= n 500))
                        [n (:score g) (boolean (s/alive? g)) (boolean (:won g))]
                        (let [ms (filter :safe (s/moves g))
                              best (when (seq ms) (apply max-key :advance ms))]
                          (if (nil? best) [n (:score g) false (boolean (:won g))]
                              (recur (s/step g (:direction best)) (inc n)))))))
          a (run) b (run)]
      (is (= a b))
      (is (true? (nth a 2)) "the cycle-safe greedy walk never dies"))))

(deftest won-when-board-fills
  (testing "a 4x4 board fills and wins"
    (let [g (assoc (s/new-game 4 4 7 15) :food [3 2])
          eat (some #(when (and (:safe %) (:eats %)) %) (s/moves g))]
      (is (some? eat) "food sits on the cycle ahead of the head")
      (is (= "RIGHT" (:direction eat)))
      (let [g2 (s/step g "RIGHT")]
        (is (= 16 (count (:body g2))))
        (is (true? (:won g2)))
        (is (nil? (:food g2)))))))
