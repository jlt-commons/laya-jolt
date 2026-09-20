# lev plays snake

The english encoder drives a snake game over raylib. Every tick the board
becomes a two-line state string, the encoder answers three typed questions
(one choice, two noul), and a deterministic safety shield turns the answer
into a move that cannot kill the snake. The model runs inside this process
through the same engine the server uses, so there is no HTTP hop and no
port involved.

This is a standalone jolt project, not part of the lev server. `deps.edn`
reaches the checkout one level up with `:local/root`, so the two build
together but stay separate: nothing here lands in lev's own `src/`.

## What you need

- jolt (Homebrew; the lev README names the minimum version)
- raylib 6 as the system shared library: `brew install raylib` on macOS,
  your distro's raylib package on Linux
- the lev checkout at `../..`, with its kernels built and one checkpoint
  prepared:

```bash
cd ../..        # the lev root
jolt kernels
jolt prepare    # ../laya -> data/ (see the lev README, "Getting the checkpoints")
```

Only the english checkpoint is used (`data/` in the lev root). If your
prepared data lives somewhere else, point `LEV_DATA` at it.

## Start lev and snake together

```bash
cd examples/snake
jolt -M:run
```

The first run compiles lev's namespaces, so give it a moment. A window
opens with the board and the game starts on its own.

There is no second process to launch: snake loads the encoder directly,
the same agent the server would load. If you also want lev's HTTP API
running next to the game, to poke the same questions yourself:

```bash
cd ../..
jolt -M:serve   # http://127.0.0.1:8080, POST /v1/systemone
```

The demo never calls it; the two just share the prepared `data/`.

## Controls

- SPACE restarts
- G toggles the safety shield
- ESC quits

## Reading the HUD

- Top line: score, length, tick count, and how many times the shield
  replaced the model's move.
- Second line: shield state, the model's proposed move, the move actually
  executed, dead-end risk, food reachability, and the milliseconds that
  decision took.
- Dark green cells: the moves the planner considers safe.
- Yellow frame on the head: the shield intervened on that tick.

Turn the shield off with G to watch the raw model. Unguarded, the first
wall-or-tail answer ends the game: the shield's job is making a wrong
answer survivable, because a fast classifier is not a planner.

## How a tick decides

1. `snake.game` computes the legal moves and, from the hamiltonian cycle
   the board is built on, which of those are safe.
2. `snake.policy` renders the compact state ("Safe route: yes. Food
   reachable through empty cells: yes.") and asks three questions: a
   choice over UP, DOWN, LEFT and RIGHT with a one-line criterion per
   direction, and two noul questions (is a safe route available, is the
   food reachable).
3. The argmax of the choice probabilities is the proposal. Guarded (the
   default), the proposal is clamped to the planner's safe directions,
   which is why the snake cannot trap itself.
4. `snake.core` steps the game and draws.

The game rules, the cycle-safety planner and the question set are ports
of the snake demo in laya-mlx, an MLX runtime for the same checkpoints.

## Tests and headless runs

```bash
jolt -M:test                          # 21 tests; one loads the real checkpoint
LEV_SNAKE_MAX_FRAMES=200 jolt -M:run  # plays 200 frames, prints a summary, exits
```

| variable | default | meaning |
|---|---|---|
| `LEV_DATA` | `../../data` | prepared checkpoint directory to load |
| `LEV_SNAKE_MAX_FRAMES` | unset | stop and exit after this many frames |
