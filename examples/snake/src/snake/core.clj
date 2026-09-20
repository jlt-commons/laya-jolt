(ns snake.core
  "lev playing snake over raylib: the encoder answers the compact question
  set every tick (snake.policy), the safety shield keeps it alive, the
  window shows the board and the model's own read of the situation.

  Run from this project with `jolt -M:run` (lev's data/ must be prepared,
  or LEV_DATA points elsewhere). Keys: SPACE restarts, G toggles the
  safety shield, ESC quits. LEV_SNAKE_MAX_FRAMES bounds the run for smoke
  tests."
  (:require [jolt.ffi :as ffi]
            [lev.agent :as ag]
            [snake.game :as s]
            [snake.policy :as p]))

;; --- the raylib surface this demo uses; Color crosses as a packed uint ---

(defn rgba [r g b a]
  (bit-or (int r) (bit-shift-left (int g) 8)
          (bit-shift-left (int b) 16) (bit-shift-left (int a) 24)))

(def BLACK (rgba 0 0 0 255))        (def RAYWHITE (rgba 245 245 245 255))
(def LIME (rgba 0 158 47 255))      (def GREEN (rgba 0 228 48 255))
(def RED (rgba 230 41 55 255))      (def GRAY (rgba 130 130 130 255))
(def YELLOW (rgba 253 249 0 255))   (def SKYBLUE (rgba 102 191 255 255))
(def ORANGE (rgba 255 161 0 255))   (def DARKGREEN (rgba 0 117 44 255))

(ffi/defcfn init-window    "InitWindow"    [:int :int :string] :void)
(ffi/defcfn set-target-fps "SetTargetFPS"  [:int] :void)
(ffi/defcfn close-window   "CloseWindow"   [] :void)
(ffi/defcfn should-close*  "WindowShouldClose" [] :int)
(ffi/defcfn begin-drawing  "BeginDrawing"  [] :void)
(ffi/defcfn end-drawing    "EndDrawing"    [] :void)
(ffi/defcfn clear-background "ClearBackground" [:uint] :void)
(ffi/defcfn draw-rectangle "DrawRectangle" [:int :int :int :int :uint] :void)
(ffi/defcfn draw-rectangle-lines "DrawRectangleLines" [:int :int :int :int :uint] :void)
(ffi/defcfn draw-text      "DrawText"      [:string :int :int :int :uint] :void)
(ffi/defcfn draw-fps       "DrawFPS"       [:int :int] :void)
(ffi/defcfn key-pressed*   "IsKeyPressed"  [:int] :int)

(def KEY-SPACE 32)
(def KEY-G 71)
(def KEY-ESCAPE 256)

(defn key-pressed? [k] (not (zero? (key-pressed* k))))
(defn should-close? [] (not (zero? (should-close*))))

;; --- layout -------------------------------------------------------------------

(def cell 30)
(def hud 64)
(def tick-frames 12)                       ; one decision every 12 frames

(defn- cell-rect [[x y]]
  [(* x cell) (+ hud (* y cell)) cell cell])

(defn- draw-game
  [{:keys [body food] :as game} d guard? ticks intervened]
  (clear-background BLACK)
  ;; the safe cells the planner would allow, under the body
  (when (s/alive? game)
    (doseq [{:keys [target safe]} (s/moves game) :when safe]
      (let [[x y w h] (cell-rect target)]
        (draw-rectangle (+ 4 x) (+ 4 y) (- w 8) (- h 8) DARKGREEN))))
  (let [[fx fy fw fh] (cell-rect food)]
    (draw-rectangle (+ 3 fx) (+ 3 fy) (- fw 6) (- fh 6) RED))
  (doseq [[i seg] (map-indexed vector body)]
    (let [[x y w h] (cell-rect seg)]
      (draw-rectangle (+ 1 x) (+ 1 y) (- w 2) (- h 2)
                      (if (zero? i) GREEN LIME))))
  ;; the head cell framed yellow when the shield intervened this tick
  (when (and d (:intervened d))
    (let [[x y w h] (cell-rect (s/head game))]
      (draw-rectangle-lines x y w h YELLOW)))
  ;; hud
  (draw-text (format "lev snake  score %d  len %d  ticks %d  interventions %d"
                     (:score game) (count body) ticks intervened)
             8 8 20 RAYWHITE)
  (draw-text (format "shield %s  model %s -> %s  risk %.2f  food %.2f  %.0f ms"
                     (if guard? "on" "OFF") (:proposed d "-") (:executed d "-")
                     (double (or (:dead-end-risk d) 0)) (double (or (:food-reachable d) 0))
                     (double (or (:inference-ms d) 0)))
             8 32 20 (if (:intervened d) YELLOW SKYBLUE))
  (when-not (s/alive? game)
    (draw-text (format "GAME OVER (%s) - SPACE to restart" (or (:death-reason game) ""))
               60 (+ hud 200) 28 ORANGE))
  (when (:won game)
    (draw-text "WON - SPACE to restart" 60 (+ hud 200) 28 ORANGE)))

(defn -main
  [& _]
  (let [max-frames (some-> (System/getenv "LEV_SNAKE_MAX_FRAMES") Integer/parseInt)
        agent (delay (ag/load-agent (or (System/getenv "LEV_DATA") "../../data")))]
    (init-window (* (:width (s/new-game)) cell)
                 (+ hud (* (:height (s/new-game)) cell))
                 "lev plays snake (system one)")
    (set-target-fps 60)
    (loop [frame 0 game (s/new-game) d nil guard? true ticks 0 intervened 0]
      (if (or (should-close?) (key-pressed? KEY-ESCAPE)
              (and max-frames (>= frame max-frames)))
        (do (close-window)
            (println (format "lev snake: %d ticks, score %d, len %d, %d interventions%s"
                             ticks (:score game) (count (:body game)) intervened
                             (if (s/alive? game) "" (str ", died: " (:death-reason game))))))
        (let [restart (or (key-pressed? KEY-SPACE) (and (:won game) false))
              guard? (if (key-pressed? KEY-G) (not guard?) guard?)
              over? (not (s/alive? game))
              [game d ticks intervened]
              (cond
                restart [(s/new-game) nil 0 0]
                (or over? (:won game)) [game d ticks intervened]
                (zero? (mod frame tick-frames))
                (let [d (try (p/decide @agent game {:guarded guard?})
                             (catch Exception e
                               (println "decision failed:" (.getMessage e)) nil))
                      d (or d (let [ms (s/moves game)]
                                {:proposed "-" :executed (or (:direction (first (filter :safe ms))) "UP")
                                 :intervened false :dead-end-risk 1.0 :food-reachable 0.0
                                 :inference-ms 0.0}))]
                  [(s/step game (:executed d)) d (inc ticks)
                   (if (:intervened d) (inc intervened) intervened)])
                :else [game d ticks intervened])]
          (begin-drawing)
          (draw-game game d guard? ticks intervened)
          (end-drawing)
          (recur (inc frame) game d guard? ticks intervened))))))
