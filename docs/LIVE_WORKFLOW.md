# From an equipped deck to a policy playing it live

The end-to-end path. Each step says what breaks if you skip it, because the
failure modes here are mostly silent — a mis-set value does not raise, it
just quietly trains or plays a different game than you think.

---

## 1. Choose the deck and set your real card levels

**`configs/decks.yaml` -> `my_deck`** — the eight cards you actually have
equipped in-game.

**`configs/card_levels.yaml`** — your real levels, including your King Tower.

Why the levels matter: `cards.yaml` holds level-1 values, and if everything
scaled uniformly the level would be irrelevant (breakpoints are ratios). Two
things break that:

- **Mixed levels.** A level-13 Hog Rider beside a level-11 Musketeer is not
  an all-level-11 deck. Every spell-kills-troop threshold in that deck moves.
- **Per-card rounding.** The ladders are ~10%/level but rounded
  independently — 59 distinct HP ladders across the roster.

Set these honestly. Guessing high teaches the policy trades you cannot make.
Levels are baked into the checkpoint, and the live bridge reads them from
there rather than from the live config, so the deployed policy always plays
the game it trained on.

---

## 2. Train that deck against the field

```bash
python -m src.agent.train --run my_deck --stage my_deck \
  --config configs/training_human.yaml
```

The `my_deck` stage pins **your** deck and lets opponents keep sampling
`ladder_top50`. That asymmetry is the point: you equip one deck in the real
game, so specializing on it is a far easier problem than generalizing over
seventy — but fixing *both* sides would overfit to a single matchup and
teach nothing about the field.

Use `configs/training_human.yaml`, not the default. The `full` tier reads the
opponent's exact elixir and is simulator-only; the live bridge refuses to run
a `full`-tier checkpoint at all.

Optionally strengthen it by distilling from a `full`-tier teacher
(`python -m src.agent.distill --teacher ...`), which is allowed because the
teacher's privilege never reaches inference.

---

## 3. Get real captures early — in parallel with training

**Do not leave this until after training.** It is the step most likely to
fail or to change the plan, and it does not depend on having a policy. The
vision pipeline is currently tested only against synthetic frames.

1. `python -m src.live --diagnose` to capture a frame and print what match
   and ready detection actually see.
2. Record an **empty arena** (a few seconds, no units) and a deploy pass:
   place each card in your deck alone and record it. This is the input to
   the label-free pipeline and it is the only manual step in it — you are
   playing the cards, not labelling anything.

   ```bash
   # harvest -> composite -> train; see README, "Training the detector
   # without labelling anything"
   python -m src.live.train_detector --manifest data/synth/manifest.json \
       --out checkpoints/detector.pt --epochs 20
   ```

3. Label a handful of real captures into `tests/fixtures/live/` — see
   `tests/live_frames.py` for the format. The test suite picks them up
   automatically and holds them to the same assertions as the synthetic
   ones. This is for *testing*, not training: a dozen frames is plenty to
   catch a broken homography or an off-by-one crop, and the detector never
   sees them.
4. If you are running the `vision.py` bar segmenter for HP (you probably
   are — the detector does not read HP), re-fit `vision.DEFAULT_TEAM_COLORS`
   and `VisionConfig` against those frames. The shipped values are starting
   points; exact tints shift with arena skin and display colour management.
   The detector itself needs no such refit — appearance randomization during
   training is what buys that.
5. Measure your actual detection error and refit `obs_noise` in
   `configs/training_human.yaml`. Those rates are **placeholders**. The
   recall reported by `train_detector` is a first estimate of `1 - p_miss`,
   but it is measured on synthetic scenes; re-measure on real captures
   before training a policy against it. Noise that is qualitatively wrong —
   a systematic homography bias modelled as zero-mean jitter — transfers
   worse than no randomization at all.
6. Once the detector runs live, turn on self-labelling
   (`src/live/autolabel.py`). It banks real frames the cycle tracker can
   name unambiguously, and retraining on synthetic plus banked is what
   closes the gap composited scenes leave.

---

## 4. Calibrate the bridge

In `configs/live_play.yaml`:

| Key | Without it |
|---|---|
| `homography_anchors` | every placement lands at the same wrong pixel |
| `elixir_bar` | the affordability mask is fabricated; the policy picks cards it cannot pay for |
| `deck` | the hand cycle drifts one card at a time |
| `checkpoint` | no policy to run |

All four are checked at load time, because each fails silently at runtime.

Measure the homography anchors as the **centres** of the six tower positions
plus the two bridge ends. Use all eight: anchors clustered in one half fit
that half and drift badly in the other, and the far-half anchors are what
pin down the perspective term.

---

## 5. Dry run, then arm

```bash
python -m src.live --config configs/live_play.yaml            # logs only
python -m src.live --config configs/live_play.yaml --armed    # taps
```

Unarmed first, and actually read the log. Compare what the policy *would*
play against what you would play. This is the cheapest place to catch a
mis-measured anchor or a deck that does not match your loadout — both look
like "the policy is bad" rather than "the calibration is wrong".

The runner degrades to the `dynamic_slots` heuristic if perception drops out
mid-match, so keep `dynamic_target` calibrated even in policy mode. Tapping a
safe card at a safe spot is a much better failure mode than acting on a
fabricated board.

---

## What the bridge does and does not know

**Real:** own hand and elixir, tower alive/dead, unit positions and teams,
unit identity where the classifier is confident.

**Approximated:** unit HP from health-bar fill (a full bar and an occluded
one are the same pixels — those are reported `hp_confident=False`), and unit
identity where the classifier abstains.

**Absent:** attack cooldowns, targeting locks, deploy timers, charge and ramp
state. No observation tier contains these, so the policy never learned to
use them and their absence costs nothing.

**Derived, never read:** opponent elixir and cycle, from
`OpponentTracker` — computed from observed play, which is what a strong
player does. It is wrong exactly when perception was wrong, which is the same
failure mode a human has.

---

## Re-benchmark after any balance change

Card levels change the game. So does a card-stat sync. Every previously
recorded win rate is then measured against different physics:

```bash
python -m scripts.rebenchmark --matches 40
```
