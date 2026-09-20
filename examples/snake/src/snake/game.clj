(ns snake.game
  "Deterministic snake rules and a separately identified cycle-safety
  planner, ported from laya-mlx's laya_mlx/snake/game.py: the board is a
  hamiltonian cycle, every cell named by its index along it. Legality is
  wall / reverse / body (the tail vacates on a non-growing step); safety
  is the planner's own judgement — a legal move that would cross or meet
  the tail along the cycle, or skip the food on the safe route, is unsafe."
  (:import (java.util Random)))

(def directions ["UP" "DOWN" "LEFT" "RIGHT"])

(def vectors {"UP" [0 -1] "DOWN" [0 1] "LEFT" [-1 0] "RIGHT" [1 0]})

(defn hamiltonian-cycle
  "Every cell once, adjacent steps, closing edge included (hamiltonian_cycle).
  Boards must be at least 4 in each dimension with at least one even."
  [width height]
  (when (or (< (min width height) 4) (and (odd? width) (odd? height)))
    (throw (ex-info "board must be >= 4 each way, with at least one even dimension"
                    {:width width :height height})))
  (if (odd? width)
    (mapv (fn [[x y]] [y x]) (hamiltonian-cycle height width))
    (let [rows (mapcat (fn [y]
                         (let [xs (if (even? y) (range 1 width) (range (dec width) 0 -1))]
                           (mapv (fn [x] [x y]) xs)))
                       (range height))]
      (into [[0 0]]
            (concat rows (mapv (fn [y] [0 y]) (range (dec height) 0 -1)))))))

(defn- spawn-food
  "rng/choice over the empty cycle cells (seeded, so replays match)."
  [{:keys [cycle rng body]}]
  (let [occupied (set body)
        empty (filter (complement occupied) cycle)]
    (if (seq empty)
      (nth empty (.nextInt ^Random rng (count empty)))
      nil)))

(defn new-game
  "([width height seed initial-length] [width height seed] [width height])
  The fresh game (SnakeGame.__init__): the body lies along the cycle
  behind the centre start."
  ([width height seed initial-length]
   (let [cycle (hamiltonian-cycle width height)
         capacity (* width height)
         indices (into {} (map-indexed (fn [i cell] [cell i]) cycle))
         start (get indices [(quot width 2) (quot height 2)])
         rng (Random. (long seed))
         body (mapv (fn [i] (nth cycle (mod (- start i) capacity)))
                    (range initial-length))
         game {:width width :height height :seed seed
               :cycle cycle :indices indices :capacity capacity
               :rng rng :initial-length initial-length
               :body body :score 0 :ticks 0
               :alive true :won false :death-reason nil :food nil}]
     (assoc game :food (spawn-food game))))
  ([width height seed] (new-game width height seed 6))
  ([width height] (new-game width height 7 6))
  ([] (new-game 24 16 7 6)))

(defn head [game] (first (:body game)))

(defn alive? [game] (boolean (:alive game)))

(defn target
  "The cell a direction moves the head to."
  [{:keys [body] :as game} direction]
  (let [[dx dy] (get vectors direction)]
    (when (and dx dy)
      (let [[hx hy] (first body)]
        [(+ hx dx) (+ hy dy)]))))

(defn legal-reason
  "\"wall\", \"reverse\", \"body\" or \"legal\" for a direction
  (legal_reason); the tail vacates its cell on a non-growing step."
  [{:keys [width height body food] :as game} direction]
  (let [cell (target game direction)]
    (cond
      (nil? cell) "unknown direction"
      (not (and (<= 0 (first cell) (dec width))
                (<= 0 (second cell) (dec height))))
      "wall"

      (= cell (second body)) "reverse"
      :else (let [occupied (set body)
                  occupied (if (= cell food)
                             occupied
                             (disj occupied (last body)))]
              (if (contains? occupied cell) "body" "legal")))))

(defn cycle-indices
  "cell -> index along the hamiltonian cycle (the game's own map)."
  [game] (:indices game))

(defn moves
  "One move info per direction (moves): direction, legal, safe, advance
  (steps along the cycle), reason, eats, target. Safe moves are the
  planner's: legal, never crossing the tail along the cycle (with the
  food exception), never skipping the food on the safe route."
  [{:keys [capacity indices body food] :as game}]
  (when (or (not (:alive game)) (:won game))
    (throw (ex-info "moves on a finished game" {:game game})))
  (let [head-index (get indices (first body))
        tail-index (get indices (last body))
        tail-distance (mod (- tail-index head-index) capacity)
        food-index (get indices food)
        food-distance (mod (- food-index head-index) capacity)]
    (mapv (fn [direction]
            (let [reason (legal-reason game direction)
                  legal (= "legal" reason)
                  tgt (target game direction)
                  advance (mod (- (get indices tgt head-index) head-index) capacity)
                  eats (= tgt food)
                  [safe reason]
                  (if (not legal)
                    [false reason]
                    (cond
                      (or (> advance tail-distance)
                          (and (= advance tail-distance) eats))
                      [false "would cross the tail"]

                      (or (zero? advance) (> advance food-distance))
                      [false "would skip the food on the safe route"]

                      :else [true reason]))]
              {:direction direction :legal legal :safe safe
               :advance advance :reason reason :eats eats
               :target tgt}))
          directions)))

(defn food-reachability
  "[food-reachable open-cells] by BFS from the head through non-blocked
  cells (food_reachability); the occupied tail is not treated as empty."
  [{:keys [width height body food]}]
  (let [blocked (disj (set body) (first body))
        walk (fn [x y] [[(inc x) y] [(dec x) y] [x (inc y)] [x (dec y)]])
        on-board? (fn [[x y]] (and (<= 0 x (dec width)) (<= 0 y (dec height))))
        bfs (loop [queue (conj clojure.lang.PersistentQueue/EMPTY (first body))
                   visited #{(first body)}]
              (if (empty? queue)
                visited
                (let [[x y] (peek queue)
                      next (filter (fn [cell]
                                     (and (on-board? cell)
                                          (not (contains? blocked cell))
                                          (not (contains? visited cell))))
                                   (walk x y))]
                  (recur (into (pop queue) next)
                         (into visited next)))))]
    [(contains? bfs food) (count bfs)]))

(defn step
  "Advance the game (SnakeGame.step): grow and respawn on food, die on an
  illegal move. Returns the updated game."
  [{:keys [capacity] :as game} direction]
  (when (or (not (:alive game)) (:won game))
    (throw (ex-info "cannot step a finished game" {:game game})))
  (when-not (contains? vectors direction)
    (throw (ex-info (str "unknown direction " direction) {:direction direction})))
  (let [game (update game :ticks inc)
        reason (legal-reason game direction)]
    (if (not= "legal" reason)
      (assoc game :alive false :death-reason reason)
      (let [tgt (target game direction)
            body (into [tgt] (:body game))]
        (if (= tgt (:food game))
          (let [score (inc (:score game))]
            (if (= (count body) capacity)
              (assoc game :body body :score score :won true :food nil)
              (let [g (assoc game :body body :score score)]
                (assoc g :food (spawn-food g)))))
          (assoc game :body (pop body)))))))

(defn snapshot
  "The plain-data view of a game (for replays and logs): cells, food,
  score, ticks, alive/won."
  [game]
  (let [{:keys [width height seed body food score ticks alive won death-reason]} game]
    {:width width :height height :seed seed
     :body (mapv identity body)
     :food (some-> food vec)
     :score score :length (count body) :ticks ticks
     :alive alive :won won :death-reason death-reason}))
