"""An empty-arena plate that builds itself from ordinary play.

Why this removes a step
-----------------------
`harvest.build_plate` takes a median over frames of an empty arena, which
means somebody has to *record* an empty arena, and then deploy each card
onto it one at a time. That is a setup ritual, and rituals do not get done.

But an empty arena is not something you have to stage. Over any stretch of a
real match, **every pixel of the playfield is background most of the time** —
units are sparse and they move. So the per-pixel median over a live match
already *is* the empty arena, and it costs nothing to keep one running.

Holding a window of frames to take an exact median would cost about 1.7 MB a
frame, so this uses the Σ-Δ estimator instead: nudge each pixel one step
toward the current frame every frame, and it converges on the per-pixel
median in constant memory. A unit standing still for a second moves the
estimate by a few levels and the estimate recovers; a unit that parks there
for a minute becomes background, which is correct — a building that has been
there a minute *is* part of the scene a new spawn should be measured against.

Volatility, and why the river needs it
--------------------------------------
Not every pixel settles. The river animates continuously, spell VFX linger,
and the HUD counts down. A plate is only trustworthy where the scene is
actually static, so alongside it this tracks how restless each pixel is, and
masks the restless ones out of foreground detection — without that the river
reads as a permanent wall of foreground and ruins every harvest near it.

Restlessness is measured as **frame-to-frame activity**, not as deviation
from the plate, and the difference matters. Deviation from the plate cannot
tell "this pixel keeps changing" from "something is standing here", so it
flags every new unit as restless and then masks it out of the very
foreground it is. It also has a blind spot: an animation that alternates
between two states lets the estimate sit on one of them, so half the frames
show no deviation at all and the median reads as settled.

Activity has neither problem. A unit standing still produces none, a unit
walking through produces a brief burst that the running median absorbs, and
anything that genuinely never stops moving produces it on every frame.

Nothing here reads game memory or network traffic; it only looks at pixels
that are already on screen. See CLAUDE.md, "On-Screen Visual Perception".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlateConfig:
    """Convergence and trust settings for the running background."""

    # Levels the estimate moves per frame once it is trusted. 1 is the
    # textbook Σ-Δ rate: slow enough that a unit pausing for a second barely
    # moves the estimate, which is the whole point of a median.
    # Work at 1/N resolution. This is the single biggest lever on the live
    # frame budget -- halving each dimension is four times less work in
    # every pass -- and detection does not need full resolution: a box two
    # pixels out is invisible on an overlay and a fraction of a tile once
    # projected.
    #
    # It is not free. Sprite alpha is cut from this mask, so at N=2 a
    # harvested silhouette has stair-stepped edges, which composites into
    # slightly worse training data. N=1 keeps full fidelity and costs about
    # four times as much per frame; raise N when the measured frame rate
    # matters more than the sprite edges.
    downsample: int = 2
    rate: int = 1
    # ...and the rate while still warming up, which has a different job.
    #
    # The plate is seeded from the first frame, so whatever was standing
    # there is seeded with it. At rate 1 that ghost takes as many frames to
    # fade as it has levels of contrast — longer than the warmup — and it
    # then becomes permanently self-protecting, because a ghost reads as
    # foreground and foreground is what the steady-state rule refuses to
    # learn from. The result is a phantom that never moves, which the
    # discovery pass would faithfully report as a building. Converging hard
    # first and carefully afterwards avoids the whole situation.
    warmup_rate: int = 8
    # Frames before the plate is offered as usable at all. Harvesting
    # against a plate that is still converging cuts units out of their own
    # afterimages.
    #
    # It is also the line between two update rules. While warming up the
    # estimate learns from every pixel, because it is admittedly wrong and
    # the fastest way to stop being wrong is to believe what it sees. After
    # that it stops learning from pixels it currently calls foreground —
    # otherwise a building that parks on a tile is quietly absorbed into the
    # scene and stops being detectable at all, which is the opposite of what
    # a building detector needs.
    warmup_frames: int = 60
    # Deviation from the estimate that counts as foreground, both for
    # `foreground()` and for deciding what the estimator refuses to learn
    # from.
    foreground_delta: float = 22.0
    # ...but "refuses" cannot mean "never", or a tower that falls stays on
    # the plate for the rest of the match. Foreground pixels are folded in
    # on one frame in this many, so anything genuinely permanent is absorbed
    # roughly this many times slower than the scene around it.
    absorb_every: int = 8
    # Frames between recomputations of the restlessness map. It is a running
    # median of frame-to-frame change, so it moves slowly by construction --
    # and taking a global median over a full-resolution frame was by far the
    # most expensive thing in the live loop. Recomputing it a couple of times
    # a second is indistinguishable in behaviour and nearly free.
    volatility_every: int = 20
    # How many times the frame's *typical* deviation a pixel may show before
    # it counts as never having settled. Relative to the frame rather than
    # absolute, so a noisy capture raises the bar for every pixel together
    # instead of condemning the whole board.
    volatility_tolerance: float = 4.0
    # ...and an absolute floor, so on a perfectly clean capture (typical
    # deviation ~0) the relative test does not condemn every pixel that
    # moves by one level.
    min_volatility: float = 6.0


class RunningPlate:
    """Per-pixel background median and volatility, updated one frame at a time.

    Feed it every capture. It is cheap — two passes over the frame, no
    history buffer — and it is the thing that turns "record an empty arena
    and deploy every card" into "play a match".
    """

    def __init__(self, config: PlateConfig | None = None):
        self.config = config or PlateConfig()
        self._plate: np.ndarray | None = None
        self._deviation: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._unstable: np.ndarray | None = None
        self._unstable_at = -1
        self.frames = 0

    def reset(self) -> None:
        self._plate = None
        self._deviation = None
        self._previous = None
        self._unstable = None
        self._unstable_at = -1
        self.frames = 0

    @property
    def ready(self) -> bool:
        return self._plate is not None and self.frames >= self.config.warmup_frames

    @property
    def plate(self) -> np.ndarray | None:
        """The current background estimate, or None before the first frame."""
        return None if self._plate is None else self._plate.astype(np.uint8)

    @property
    def volatility(self) -> np.ndarray | None:
        """Per-pixel running median of frame-to-frame change."""
        return None if self._deviation is None else self._deviation.copy()

    @property
    def scale(self) -> int:
        """Pixels of the original frame per pixel of the plate."""
        return max(1, self.config.downsample)

    def _reduce(self, frame) -> np.ndarray:
        """The frame at plate resolution.

        Strided slicing rather than an interpolating resize: point sampling
        a background estimate is exactly as valid as averaging it, and the
        resize would cost more than everything it feeds.
        """
        step = self.scale
        return np.asarray(frame)[::step, ::step, :3].astype(np.int16)

    def upsample(self, mask: np.ndarray) -> np.ndarray:
        """A plate-resolution mask back at frame resolution."""
        step = self.scale
        return mask if step == 1 else np.repeat(np.repeat(mask, step, 0), step, 1)

    def update(self, frame) -> None:
        """Fold one capture into the estimate."""
        array = self._reduce(frame)
        if self._plate is None or self._plate.shape != array.shape:
            # Seeding from the frame rather than from zero: the estimator
            # only has to travel where this frame held a unit, instead of
            # climbing from black across the whole board.
            self._plate = array.copy()
            self._deviation = np.zeros(array.shape, np.int16)
            self._previous = array.copy()
            self.frames = 1
            return

        cfg = self.config
        # Derive both the foreground test and the update step from one
        # subtraction. At full resolution each extra pass over the frame is
        # a measurable slice of the per-frame budget.
        diff = array - self._plate
        foreground = np.abs(diff).max(axis=2) >= cfg.foreground_delta

        warming = self.frames < cfg.warmup_frames
        # An absorb frame suspends the foreground rule, so that anything
        # genuinely permanent — a tower that has fallen, a building that has
        # stood for a minute — eventually joins the scene instead of being
        # protected from it forever.
        absorbing = not warming and self.frames % cfg.absorb_every == 0
        learn_everywhere = warming or absorbing

        rate = cfg.warmup_rate if warming else cfg.rate
        step = np.sign(diff).astype(np.int16) * rate
        if learn_everywhere:
            self._plate = np.clip(self._plate + step, 0, 255)
        else:
            self._plate = np.clip(
                self._plate + step * (~foreground)[:, :, None], 0, 255)

        # Restlessness, as the running median of frame-to-frame change.
        # Unconditional: unlike deviation from the plate, activity does not
        # confuse "keeps changing" with "something is standing here", so
        # there is nothing to gate it on. A unit that stops moving stops
        # contributing; the river never does.
        activity = np.abs(array - self._previous).astype(np.int16)
        self._deviation = np.clip(
            self._deviation + np.sign(activity - self._deviation).astype(np.int16),
            0, 255)
        self._previous = array.copy()
        self.frames += 1

    def unstable(self) -> np.ndarray | None:
        """Boolean map of pixels that never stop changing.

        Returned per pixel rather than per channel: a pixel is unusable if
        *any* of its channels is restless, and asking a caller to reduce it
        every frame would only invite them to do it differently.
        """
        if self._deviation is None:
            return None
        cfg = self.config
        fresh = self.frames - self._unstable_at
        if self._unstable is not None and fresh < max(1, cfg.volatility_every):
            return self._unstable
        worst = self._deviation.max(axis=2)
        typical = float(np.median(worst))
        threshold = max(cfg.min_volatility, cfg.volatility_tolerance * typical)
        self._unstable = worst > threshold
        self._unstable_at = self.frames
        return self._unstable

    def foreground(self, frame, min_delta: float | None = None) -> np.ndarray | None:
        """What is on the arena now that is not part of the settled scene.

        Returned at **plate resolution**, not frame resolution — see
        `PlateConfig.downsample`. Callers that need frame coordinates
        multiply by `scale`, and callers that need a frame-sized mask pass it
        through `upsample`.

        None until the plate has warmed up — an honest "I cannot tell yet",
        rather than a mask full of afterimages that looks like a busy board.
        """
        if not self.ready:
            return None
        array = self._reduce(frame)
        if array.shape != self._plate.shape:
            return None
        delta = np.abs(array - self._plate).max(axis=2)
        mask = delta >= (self.config.foreground_delta if min_delta is None else min_delta)
        restless = self.unstable()
        if restless is not None:
            mask &= ~restless
        return mask
