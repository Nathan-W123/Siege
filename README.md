# CR Bot

A reinforcement learning agent that learns to play Clash Royale entirely inside a custom battle simulator. This is an educational RL project — not a real-game bot.

**Current phase:** simulator, scripted opponents, eval reporting, and training infrastructure. The PPO agent is next.

---

## What it does

| Component | Status | Description |
|-----------|--------|-------------|
| Battle simulator | ✅ | Config-driven engine (`src/simulator/`) |
| UC-tier scripted bots | ✅ | Rusher, control, siege, beatdown heuristics |
| Deck pools & generation | ✅ | Ladder meta pool, adaptive builder, focused rotation |
| **Training reports** | ✅ | W/L, win rate, matchups, card usage, TensorBoard |
| Heroes & evolutions | ✅ | Config-driven; abilities + deck rules |
| PPO agent | ✅ | Masked factored policy + league self-play (`src/agent/`) |
| Observation tiers | ✅ | `full` / `human` / `restricted` (`src/agent/obs_layout.py`) |
| Domain randomization | ✅ | Degraded enemy detections for training (`src/agent/obs_noise.py`) |
| Teacher→student distillation | ✅ | Privileged teacher, live-legal student (`src/agent/distill.py`) |
| Opponent tracker | ✅ | Derives opponent elixir + cycle from observed play (`src/agent/opponent_tracker.py`) |
| Arena perception | ✅ | Homography, learned entity detector, spell events (`src/live/`) |
| Label-free training data | ✅ | Sprite harvest, synthetic scenes, self-labelling (`src/live/`) |
| Inference-time search | ✅ | Sim-only rollout search (`src/agent/search.py`) |
| Population-based training | ✅ | Hyperparameter search on the frozen benchmark (`src/agent/pbt.py`) |
| 3D network viewer | ✅ | Live WebGL view of the policy while it trains or plays (`src/viz/`) |
| Card levels | ✅ | Train at your real in-game levels (`configs/card_levels.yaml`) |
| **Policy-driven live play** | ✅ | A trained checkpoint drives the bridge (`src/live/bridge.py`) |

---

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip

### Live Windows desktop bridge

`src.live` can watch an already-running Clash Royale match on this Windows
desktop and play a calibrated preset deck. It does not start matchmaking. The
live bridge is separate from training because it only sees screen pixels, not
simulator state. Keep the game window visible and unobscured.

Create `configs/live_play.yaml` from
[`configs/live_play.example.yaml`](configs/live_play.example.yaml), then set
the coordinates, `reference_size` used when recording them, preset deck,
opening hand, and draw order. The example uses `desktop_capture: window`,
which finds the visible window whose title contains `window_title` and scales
coordinates to that window's current client size. This means you do not need
to know or preserve its desktop position or size. If your launcher title does
not include `Clash Royale`, change `window_title` to a distinctive part of its
title. First verify recognition with:

```bash
python -m src.live --config configs/live_play.yaml
```

That command only logs proposed plays. Once match detection and coordinates
are calibrated, pass `--armed` to permit desktop clicks:

```bash
python -m src.live --config configs/live_play.yaml --armed
```

To use the prior Android bridge instead, set `transport: adb` and supply an
`adb_path` (and optionally `device_serial`) in the same configuration.

By default, `decision_mode: dynamic_slots` re-reads the four visible card
slots on every capture rather than using a fixed deck or draw cycle. It uses
the first affordable slot in `slot_priority` and deploys it at
`dynamic_target`; it does not identify card artwork, so use a safe target and
do not expect spell-aware placement. Set `decision_mode: known_deck` only for
the legacy calibrated deck-cycle behavior.

### Install

```bash
cd "CR Bot"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -q
```

### Run benchmark + print report

Pits a bot against the **frozen benchmark suite** (rusher, control, siege, beatdown) and prints a full report:

```bash
python -m src.eval --agent-bot control --agent-deck control --matches 20
```

Example output:

```
=== Training Report: benchmark_control ===
Matches: 80  |  W 42  L 35  D 3
Win rate: 54.5%  |  W/L ratio: 1.20
Avg crowns: 1.12 for  0.98 against
Avg match: 145.3s  |  Elixir spent: 38.2  leaked: 1.40
Card usage entropy: 2.01 nats

--- vs Opponent Deck ---
  beatdown          12W    8L  WR  60.0%  W/L 1.50
  control           10W   10L  WR  50.0%  W/L 1.00
  ...
```

Export JSON or log to TensorBoard:

```bash
python -m src.eval --agent-bot rusher --agent-deck rusher --matches 50 \
  --json runs/report.json \
  --tensorboard runs/tb --step 100000
tensorboard --logdir runs/tb
```

---

## Start training

Activate the venv, then launch PPO from the first curriculum stage (`one_lane`):

```bash
.venv\Scripts\activate
python -m src.agent.train --run my_run --stage one_lane --seed 0
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--run` | `run1` | Run name; logs go to `runs/<run>/`, checkpoints to `checkpoints/<run>/` |
| `--stage` | first stage | Curriculum stage (`one_lane`, `full_arena`, `full_pool`, …) |
| `--auto` | off | Run all curriculum stages in sequence |
| `--steps` | `ppo.total_steps` (3M) | Per-stage environment-step budget |
| `--n-envs` | `8` | Parallel envs (see `configs/training.yaml`) |
| `--bc-init` | — | Warm-start from a BC checkpoint (`python -m src.agent.bc`) |
| `--resume` | — | Continue from a saved `.pt` checkpoint |

Quick smoke run (~30s):

```bash
python -m src.agent.train --run smoke --stage one_lane --steps 4096 --n-envs 2 --seed 0
```

Monitor training:

```bash
tensorboard --logdir runs/my_run/tb
```

Outputs:

- `runs/<run>/train_log.csv` — per-rollout metrics (win rate, entropy, losses)
- `runs/<run>/eval_log.csv` — periodic eval vs scripted bots
- `runs/<run>/tb/` — TensorBoard scalars
- `runs/<run>/latest.pt` and `final.pt` — policy checkpoints
- `checkpoints/<run>/pool/` — league snapshots for self-play stages

Optional BC bootstrap before PPO:

```bash
python -m src.agent.bc --matches 150 --out checkpoints/bc_init.pt
python -m src.agent.train --run my_run --stage one_lane --bc-init checkpoints/bc_init.pt
```

---

## Observation tiers

What the policy is allowed to read is a scope decision, not a tuning knob —
see CLAUDE.md, "On-Screen Visual Perception". Pick one with a training
config; the tier is stored in the checkpoint.

| Tier | Spatial grid | Own hand/elixir | Opponent elixir | Config | Live-legal? |
|------|--------------|-----------------|-----------------|--------|-------------|
| `full` | real | yes | **yes** | `configs/training.yaml` | No — simulator/critic only |
| `human` | real | yes | no | `configs/training_human.yaml` | **Yes — the live-play target** |
| `restricted` | zero-filled | yes | no | `configs/training_restricted.yaml` | Yes (fallback: no vision) |

```bash
python -m src.agent.train --run human1 --config configs/training_human.yaml --stage full_pool
```

The scalar widths are *not* ordered by tier (`restricted` is 18 wide,
`full` is 17): restricted drops opponent elixir but adds four per-tower alive
flags. Always go through `obs_layout.scalar_dim_for`. The old
`use_spatial: true|false` key still works as a tier alias so existing
checkpoints keep loading.

### Domain randomization

`human`/`full` training observations can be degraded to resemble real
detections — positional jitter (applied in tile space, before grid binning),
missed detections, false positives, identity confusion, occlusion, and
capture lag. Enemy entities only; own units, own hand/elixir, and all towers
are read reliably and are never perturbed. Configure under `obs_noise:` (see
`configs/training_human.yaml`). Applied during training only, so benchmark
numbers stay comparable.

### Teacher → student distillation

Train a strong `full`-tier teacher, then distil it into a `human`-tier
student on states the **student** visits, with the KL weight annealing to
zero:

```bash
python -m src.agent.distill --teacher checkpoints/full_final.pt \
  --run human_distill --stage full_pool --config configs/training_human.yaml
```

### Set-based unit encoder (ablation)

`network.use_set_encoder: true` swaps the CNN-over-grid for a
permutation-invariant deep-sets encoder over the entity list, keeping exact
positions and per-unit identity that the 2x2-tile grid throws away. Compare
the two on the frozen benchmark, not on self-play win rate.

### Inference-time search (simulator only)

`src.agent.search.SearchBot` wraps a policy with rollout search over its
top-k masked actions. Too slow for the 0.5s live cadence; use it for a
stronger sim agent and as a distillation teacher.

### Population-based training

`src.agent.pbt` searches PPO hyperparameters and reward weights. Fitness
comes from the frozen benchmark only — `PBTPopulation.record` rejects any
other source, because self-play win rate trends to 50% by construction.
Needs many concurrent runs to be worthwhile.

---

## Training reports

Reports track what the agent (or stand-in bot) learns over time.

### Metrics collected per match

- **Win / loss / draw** and **win rate**
- **W/L ratio** (wins ÷ losses)
- **Crowns** scored vs conceded
- **Match duration** and **elixir** spent / leaked
- **Card usage** counts and **entropy** (strategy diversity)
- Breakdowns by **opponent deck**, **opponent bot**, and **agent deck**
- **Matchup matrix** (your deck × their deck)
- **Weak matchups** flagged when WR < 50% over 3+ games

### Using the reporter in code

```python
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.training.session import run_episodes, run_focused_training

catalog = DeckCatalog()
bot = get_bot("rusher", catalog=catalog, deck_name="rusher")

# Random ladder-meta opponents
reporter = run_episodes(bot, n_episodes=100, stage="full_pool", seed=0)

# Focused rotation: beat each ladder deck in sequence, adaptive agent deck
focused = run_focused_training(bot, max_episodes=500, stage="focused_ladder", seed=0)
print(focused.summary())
focused.export_json("runs/focused_training.json")
```

When the RL agent lands, call `TrainingReporter.record()` or `record_report()` after each episode — same API.

### What to watch during training

| Signal | Good sign | Bad sign |
|--------|-----------|----------|
| Win rate vs benchmark bots | Trending up | Flat or falling |
| Self-play win rate | ~50% (expected) | Don't use as sole metric |
| Card entropy | Stable or rising | Collapsing → mode collapse |
| Per-deck WR | Balanced across archetypes | One deck at 90%, others at 30% |

---

## Deck system

### Ladder meta opponents (`configs/ladder_decks.yaml`)

70 curated Ultimate Champion decks sourced from RoyaleAPI TopRanked (7d) and Season 81 leaderboard data (March–April 2026). Each deck has a name, archetype tag, and 8 cards. The `ladder_top50` pool is auto-built from this file and is the default opponent pool in `configs/training.yaml`.

### Focused rotation curriculum

`run_focused_training()` plays matches against **one fixed ladder deck at a time**. The agent advances to the next deck after:

- **X wins** vs the current deck (`focused_rotation.wins_per_deck`), or
- **min matches** with per-deck win rate above `min_win_rate_per_deck`

After cycling through all ladder decks, training stops when overall win rate reaches **`target_win_rate` (Y)**; otherwise the cycle restarts.

Config (`configs/training.yaml`):

```yaml
focused_rotation:
  wins_per_deck: 5
  target_win_rate: 0.65
  min_matches_per_deck: 10
  min_win_rate_per_deck: 0.55
```

### Adaptive agent deck builder

During focused training the agent does **not** use a fixed deck. `AdaptiveDeckBuilder` tracks per-card performance (win rate when played, crowns, elixir efficiency) and rebuilds an 8-card deck by picking the best card in each role category (`configs/card_categories.yaml`):

- win condition → spell → building → cycle → support → swarm → air → heavy
- Max 1 hero and 1 evolution per deck

Rebuild triggers (`adaptive_deck` in `training.yaml`):

- Every `rebuild_every_matches` games, or
- When win rate plateaus over `plateau_window` matches

Card scores are exported in JSON reports under `card_scores`.

---

```
configs/
  cards.yaml          # Card stats (balance lives here)
  ladder_decks.yaml   # Top UC ladder meta decks (~70)
  card_categories.yaml # Roles for adaptive deck builder
  decks.yaml          # Named decks + pools
  deck_templates.yaml # Procedural deck generation
  arena.yaml          # Arena geometry & clock
  curriculum.yaml     # Training stages (incl. focused_ladder)
  training.yaml       # PPO hyperparams, focused rotation, adaptive deck
  eval.yaml           # Frozen benchmark opponents

src/
  simulator/          # Battle engine
  bots/               # Scripted UC-tier opponents
  decks/              # Catalog, sampling, adaptive builder, deck search
  training/           # Opponent sampling, focused curriculum, sessions
  agent/              # Gym env adapter, PPO, policy network, BC, league
  eval/               # Reports, benchmarks, TensorBoard

tests/                # Unit + integration tests
```

---

## Training pipeline

1. **Imitation bootstrap (optional)** — `python -m src.agent.bc` from scripted bot rollouts
2. **PPO self-play** — `python -m src.agent.train`; league of past checkpoints in self-play stages
3. **Eval discipline** — always benchmark against frozen bots in `configs/eval.yaml`

Opponents train at **Ultimate Champion** tier: elixir discipline, counter-play, spell value, archetype offense.

Curriculum stages (`configs/curriculum.yaml`):

1. `one_lane` — single lane, small deck, UC scripted bots
2. `full_arena` — full map, mirror deck, self-play mix
3. `full_pool` — both sides sample from ladder meta pool
4. `focused_ladder` — focused rotation vs one ladder deck at a time + adaptive agent deck

---

## Heroes & evolutions

The simulator supports CR-style **heroes** and **evolutions** via config:

| Feature | Config | Rules |
|---------|--------|-------|
| **Heroes** | `configs/heroes.yaml` | Max 1 per deck; ability charges on arena, auto-fires at full charge |
| **Evolutions** | `configs/evolutions.yaml` | Max 1 per deck; loaded as `{base}_evo` cards with stat overrides |

**Hero abilities** (simplified): `dash` (leap + burst damage), `spawn` (summon troops), `damage_buff` (temporary attack boost).

**Evolution examples:** `knight_evo`, `hog_rider_evo`, `skeletons_evo`, `fireball_evo`, etc.

Deck validation enforces: no duplicate base+evo pair, one hero, one evolution. Archetype decks (`rusher`, `control`, etc.) already include a hero and evolution each.

---

## Configuration

Key files — edit YAML, not code, for balance and training tweaks:

- **`configs/cards.yaml`** — HP, damage, cost, range
- **`configs/heroes.yaml`** — hero troops and abilities
- **`configs/evolutions.yaml`** — evolution stat overrides
- **`configs/decks.yaml`** — deck lists and pools
- **`configs/ladder_decks.yaml`** — UC meta opponent decks
- **`configs/card_categories.yaml`** — adaptive builder role mapping
- **`configs/training.yaml`** — PPO settings, `focused_rotation`, `adaptive_deck`
- **`configs/eval.yaml`** — benchmark roster (never delete opponents)

---

## Playing your own deck live

`docs/LIVE_WORKFLOW.md` is the end-to-end path: pick the deck, set your real
card levels, train it against the ladder field, calibrate the bridge, dry-run,
then arm.

```bash
# 1. edit configs/decks.yaml -> my_deck, and configs/card_levels.yaml
# 2. train it against the field (your deck pinned, opponents sampled)
python -m src.agent.train --run my_deck --stage my_deck   --config configs/training_human.yaml
# 3. calibrate configs/live_play.yaml (decision_mode: policy)
# 4. watch it decide without tapping, then arm
python -m src.live --config configs/live_play.yaml
python -m src.live --config configs/live_play.yaml --armed
```

The bridge reconstructs a *shadow engine* from perception
(`src/live/bridge.py`) and reuses the simulator's own observation encoder and
action masking, so the policy sees exactly the action space it trained on
rather than a re-implementation that could drift. A `full`-tier checkpoint is
refused — it reads opponent elixir, which no player can see. If perception
drops out mid-match the runner degrades to the heuristic rather than acting
on a fabricated board.

### Card levels

`configs/cards.yaml` holds level-1 values, and `configs/card_levels.yaml`
scales them to the levels you actually own — using the real per-level ladders,
not a 10%/level approximation. It matters for two reasons: **mixed levels**
move every breakpoint in a deck, and per-card **rounding** means even a
uniform shift is not exactly a no-op (59 distinct HP ladders across the
roster). Levels are recorded in the checkpoint, and the live bridge reads
them from there, so the deployed policy always plays the game it trained on.

---

## Arena perception

Reading the rendered arena is in scope: a human sees troops by looking at the
display, so a vision system that reads them replicates ordinary perception.
Memory reading and packet inspection remain out of scope — see "Live-play
scope" below and CLAUDE.md.

- **`src/live/homography.py`** — pixel↔tile mapping. The arena is drawn in
  perspective, so an affine scale is wrong; calibrate a full homography from
  the six tower centres plus the two bridge ends via `homography_anchors:` in
  `configs/live_play.yaml`. Provides both directions: pixel→tile for
  perception, tile→pixel so a trained policy can deploy anywhere instead of
  at one fixed configured point. Degenerate or inconsistent anchors are
  rejected at config load, not mid-session.
- **`src/live/detector.py`** — the learned entity detector, and the main
  path. Anchor-free (CenterNet-style) so decode cost does not grow with how
  busy the board is, and small enough to share the ~0.5s decision budget
  with a policy forward pass. Card, team and kind are **separate heads**: a
  card name among a hundred is hard, while friendly-or-hostile and
  troop-or-building stay accurate long after identity gives up. Below the
  identity threshold the card blanks and the kind, team and position still
  come through — which matters because kind decides which observation
  channel the entity lands in at all.
- **`src/live/spells.py`** — spell casts, from frame-to-frame change. A
  spell is not an object: no health bar, no team tint, nothing left behind,
  so no per-frame detector can find one at any threshold. What it has is a
  temporal signature — a bloom that grows, peaks and collapses in under a
  second, in place, where a walking troop's change region translates and
  holds its area. Position and radius are measured off the VFX and projected
  through the homography, so the radius is perspective-corrected.
- **`src/live/vision.py`** — team-tinted health-bar segmentation, connected
  components, and HP from bar fill. Retained because bar fill is the one
  thing the detector does **not** read: it is not trained on HP, and a
  fabricated fraction would flow into the trade arithmetic the policy
  defends with. *Known fidelity loss:* a full bar has no visible remainder,
  so "full HP" and "bar occluded" are the same pixels; those detections are
  reported at full HP with `hp_confident=False`.
- **`src/live/identify.py`** — template matching over the deck prior. The
  original identity path, kept as a no-training fallback for when no
  detector checkpoint has been trained yet. Its bounding cost — a template
  library is specific to one skin at one resolution, and opponent cards need
  hand labelling — is what the pipeline below removes.
- **`src/agent/opponent_tracker.py`** — opponent elixir and hand, *derived*
  from observed play (start value + known regen + observed card costs, and
  the deterministic 8-card cycle) rather than read. It consumes only what
  perception reported and never touches engine state, so it is wrong exactly
  when perception was wrong — the same failure mode a human has.

Detector templates, sprites, checkpoints and annotated frames are not
bundled: they are display-specific, and shipping someone else's would be
worse than having none. `tests/live_frames.py` documents the fixture format
and generates synthetic frames; drop real annotated captures into
`tests/fixtures/live/` and the test suite picks them up automatically.

### Training the detector without labelling anything

The detector needs labelled data and nobody labels any. Four steps, each
producing labels as a byproduct of something you were doing anyway.

**1. Harvest** (`src/live/harvest.py`). Deploy a known card onto an empty
arena and difference the frame against a plate of that same arena. What is
left is that card's pixels with a pixel-exact alpha, named because you chose
it; stepping the frames yields every animation pose and facing. The plate is
a median, not a mean — the arena is never still, and a mean smears the river
animation into the plate where it subtracts forever after as a faint
permanent sprite.

The assumption worth naming: you never need the *opponent* to show you a
card to learn what it looks like. A card's sprite is the same sprite whoever
plays it, and only the health bar is team-tinted — one hue rotation away
(`synth.recolor_team`).

**With one real limit.** You play from the bottom, so your troops walk up
the board and show their backs while the opponent's walk down and show their
fronts, and those are separate art — no flip turns one into the other. A
retinted friendly sprite is therefore a truthful enemy example of *where*,
*whose*, *how big* and *what kind*, and a false one of *which card*. Those
examples are marked `identity_supervised=False`, train a trailing
"entity, card unknown" heatmap channel instead of a card channel, and have
the card channels' loss masked where they sit — so the model is never taught
that an enemy Knight looks like the back of one. Enemy card identity arrives
in step 4 instead, from real frames facing the right way.

**2. Composite** (`src/live/synth.py`). Harvested sprites were alone on an
empty board, so paste them back at random tiles in random overlapping
combinations. Because it placed them, it knows every box, class, kind and
team exactly. Sprites are rescaled by the perspective ratio between where
they were harvested and where they land — the homography knows that factor —
and scenes paint back to front so nearer units occlude farther ones, the
only occlusion order the game produces. Occlusion is *measured* with an
owner map, and a sprite left too buried loses its annotation rather than
teaching the detector to hallucinate units behind units.

```bash
python -m src.live.train_detector --manifest data/synth/manifest.json \
    --out checkpoints/detector.pt --epochs 20
```

**3. Watch recall, not loss.** Detection losses fall smoothly while the
model still finds nothing — background cells dominate them. Training reports
centre recall and identity accuracy, which are also the two failure modes in
play: a miss is a blind spot, a wrong name is a bad trade, and only the
second is recoverable.

**4. Self-label** (`src/live/autolabel.py`). Synthetic scenes are wrong in
ways that only show on a real frame: real pushes clump along lanes, harvest
artifacts repeat thousands of times, nothing is ever drawn mid-death.
`OpponentTracker` fixes this for free — by the time a card is played it
knows the deck, the hand, and what they can afford. Stack that against the
detector's kind and the number of bodies that appeared at once (three
together is a card with `count == 3`) and the answer is usually forced. One
survivor is a deduction, not a guess, and the crop under it is a labelled
example on a real frame in your skin at your resolution. More than one
survivor banks nothing.

Banked frames use the same manifest format as the synthetic ones, so a
training run mixes them by pointing at both. Play, bank, retrain, play — a
new arena skin or a new season becomes a night of self-labelling rather than
an afternoon of anyone's time.

---

## 3D network viewer

`src/viz/` serves a WebGL view of the policy network — the real one, derived
from the module tree, so it changes when `use_set_encoder`, `use_recurrence`
or `critic_tier` change. A terminal pane underneath carries the same output
the process prints.

```bash
python -m src.viz                                    # fresh net, watch it learn
python -m src.viz --checkpoint checkpoints/full_pool_60m.pt --mode live
```

Then open <http://localhost:8770>. Drag to orbit, right-drag or shift-drag to
pan, wheel to zoom, `WASD`/`QE` to fly, `R` to reframe, `F` to toggle labels.

To watch a **real** run rather than the self-contained sources above, attach
the viewer to the process doing the work:

```bash
python -m src.agent.train --run r1 --stage full_pool --viz-port 8770
python -m src.live --config configs/live_play.yaml --checkpoint ckpt.pt --viz-port 8770
```

Two signals drive the picture, and they mean different things:

| Signal | Shows | Drives |
|--------|-------|--------|
| **activation** | what fired on the last forward pass | travelling pulses along edges, node flare |
| **reveal** | how far each unit's weights have moved *since the viewer attached* | node size and opacity |

**The network does not grow while training** — the architecture is fixed at
construction. What grows is how much of it has demonstrably *changed*:
`reveal` is monotonic, so an untrained net starts as a faint skeleton and
fills in, and a unit whose weights never move stays dark. Because movement is
measured from attach, a loaded checkpoint starts at zero; `maturity`
(weight-norm dispersion) is sent alongside so a trained network still renders
as one.

Two deliberate simplifications, both stated in the UI: layers wider than 48
units are drawn with 48 evenly spaced **real** units, and edges are the 3
strongest incoming weights per drawn node — a 256x256 layer pair is 65k lines
and renders as a grey slab.

The server binds to loopback and is off unless asked for; with no viewer
attached the telemetry calls in the training loop are no-ops. Add `?snapshot`
to the URL to hold the last frame and drop the stream, which is what makes
the page capturable by screenshot tools.

---

## Simulator fidelity

`docs/SIM_FIDELITY.md` is the audit of where the simulation diverges from
real Clash Royale, what was fixed (charge mechanics, inferno ramp-up, stun
resetting wind-ups, death spawns/damage, tunnelling, obstacle steering,
continuous tornado pull, persistent rage zones), and what was deliberately
left alone and why.

### Card stats

`configs/cards.yaml` holds **level-1 base Clash Royale values** — the same
table `configs/arena.yaml`'s towers come from. Troops all scale on one
per-level curve, so level-1 values preserve every real breakpoint; there is
nothing to rescale.

Keep it honest with:

```bash
python -m scripts.sync_card_stats            # per-card diff vs the real table
python -m scripts.sync_card_stats --apply    # rewrite the syncable fields
python -m scripts.sync_card_stats --refresh  # re-download the reference
```

The reference (`configs/reference_card_stats.json`) is a vendored
distillation of [RoyaleAPI's `cr-api-data`](https://github.com/RoyaleAPI/cr-api-data)
mirror of Supercell's card table. The tool only rewrites fields whose
schemas align cleanly (`hp`, `damage`, `hit_speed`, `speed`, `count` — the
ones breakpoints depend on) and reports the rest rather than guessing.

**Any `--apply` is a balance change.** Re-establish the baseline afterwards,
because every earlier win rate was measured against different physics:

```bash
python -m scripts.rebenchmark --matches 40
```

---

## Live-play scope

The optional live bridge uses screen capture and calibrated clicks only. It
does not start matchmaking, read game memory, or access network traffic.

---

## License

Personal / educational project.
