"""Spell detection from frame-to-frame change.

Why spells need their own detector
----------------------------------
`vision.py` finds units by segmenting their health bars, and `identify.py`
names them from the sprite underneath. Neither can ever see a spell, at any
threshold, because **a spell is not an object**. A fireball is roughly half a
second of expanding VFX with no health bar, no team tint, and nothing left
behind; by the time anything persistent exists, the spell has already
resolved. Tuning the unit detector to find one is not a matter of loosening a
constant — there is no per-frame appearance to find.

What a spell *does* have is a temporal signature, and it is a distinctive
one: a sudden bloom of high-contrast change over a large, compact area, which
grows for a few frames, peaks, and collapses. Ordinary arena motion does not
look like that. A troop walking produces a change region that is small,
roughly constant in area, and **translating**; a spell's region is large,
**stationary**, and transient. Separating the two is what this module does,
and it needs no training and no labels to do it.

What is measured, and what is not
---------------------------------
Position and radius are measured off the VFX footprint and projected through
the same homography the unit detector uses, so they are in arena tiles and
carry the perspective correction. Damage is *not visible at any radius* —
see `resolve_identity`, which recovers identity from the measured radius
narrowed by what the opponent can still be holding, rather than from pixels.

Latency, stated plainly
-----------------------
An event is emitted when its footprint **stops growing**, not when it first
appears. Emitting at onset would be one or two frames sooner but would
measure the radius of a spell that has not finished expanding, and radius is
the only handle `resolve_identity` has. The cost is real: at 20 fps that is
roughly 100-200 ms of delay against a spell that takes about a second to
resolve. It is the right trade while radius carries the identity, and the
wrong one if a future detector names spells from appearance instead.

Nothing here reads game memory or network traffic; it only looks at pixels
that are already on screen. See CLAUDE.md, "On-Screen Visual Perception".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.live.vision import connected_components

# ----------------------------------------------------------------- tuning


@dataclass(frozen=True)
class SpellConfig:
    """Thresholds for the change detector.

    Unlike the hue windows in `vision.py`, these are not display-colour
    dependent — they are magnitudes of *change* and ratios of area, which
    survive a skin change far better than an absolute tint does. They still
    want checking against real captures before a session is trusted.
    """

    # Work at 1/N resolution. The footprint of interest is tens of pixels
    # across, so a quarter-scale image loses nothing that matters and cuts
    # the component search by 16x.
    downsample: int = 4
    # Luma change (0-255) that counts as "this pixel changed".
    min_delta: float = 28.0
    # A change region smaller than this is a troop moving, not a spell.
    # In downsampled pixels.
    min_area: int = 40
    # ...and one larger than this fraction of the frame is a scene
    # transition (match start, a screen wipe, the crown-tower cutscene), not
    # a spell. Emitting on those would put a fabricated fireball on the
    # arena at exactly the moment the board is least readable.
    max_area_fraction: float = 0.35
    # A spell blooms in place. If the centroid travels more than this many
    # downsampled pixels per frame, it is something moving across the arena.
    max_drift_px: float = 6.0
    # A track older than this was never a spell — spells are brief.
    max_duration: float = 2.0
    # ...and one that vanishes faster than this is single-frame noise.
    min_duration: float = 0.05
    # Frames a track may go unmatched before it is closed out.
    max_gap_frames: int = 2
    # Ratio of peak area to the area at first sight. A spell grows; a
    # persistent object that merely got brighter does not.
    min_growth: float = 1.35


@dataclass(frozen=True)
class SpellEvent:
    """One detected spell cast, in arena coordinates.

    `card` is empty here by construction: this detector measures a footprint
    and never claims an identity. `resolve_identity` is the separate step
    that names it, because naming needs the opponent's card cycle and not
    more pixels.
    """

    tile_x: float
    tile_y: float
    radius: float          # tiles
    at: float              # seconds, on whatever clock `observe` was fed
    peak_area: int         # downsampled pixels, for diagnostics
    growth: float          # peak area / onset area
    confidence: float


@dataclass
class _Track:
    """A change region followed across frames."""

    cx: float
    cy: float
    area: int
    onset_area: int
    peak_area: int
    first_seen: float
    last_seen: float
    peak_cx: float
    peak_cy: float
    peak_extent: float     # half-width in downsampled px at peak
    missed: int = 0
    shrinking: bool = False
    emitted: bool = False


def _luma(array: np.ndarray, step: int) -> np.ndarray:
    """Downsampled luma plane. Strided slicing, not an interpolating resize:
    a spell footprint is tens of pixels across, so point-sampling it costs
    nothing and avoids pulling in a resampling dependency."""
    small = array[::step, ::step, :3].astype(np.float32)
    return 0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]


class SpellWatcher:
    """Stateful frame-difference detector. Feed it every capture.

    Stateful because the signal *is* the state — a single frame cannot show
    that something bloomed. Callers that drop frames (a stalled capture, a
    skipped decision tick) should keep feeding it anyway; gaps degrade the
    growth measurement but do not corrupt it.
    """

    def __init__(self, homography=None, arena=None, config: SpellConfig | None = None):
        self.config = config or SpellConfig()
        self.homography = homography
        self.arena = arena
        self._prev: np.ndarray | None = None
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        """Drop all state. Call between matches — a track carried across a
        match boundary would emit against the new board."""
        self._prev = None
        self._tracks = []

    # ------------------------------------------------------------- stepping

    def observe(self, image: Image.Image | np.ndarray, now: float,
                homography=None) -> list[SpellEvent]:
        """Feed one frame; get back any spells that finished expanding.

        Returns an empty list on the vast majority of frames. That is the
        expected case, not a failure to detect.

        `homography` overrides the one held on the watcher, which is how a
        caller whose capture is rebased per frame (the window moved, or the
        client was resized) keeps projection correct without rebuilding the
        watcher and losing its in-flight tracks.
        """
        cfg = self.config
        if homography is not None:
            self.homography = homography
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            return []
        current = _luma(array, cfg.downsample)

        previous, self._prev = self._prev, current
        if previous is None or previous.shape != current.shape:
            # First frame, or the capture size changed under us. Either way
            # there is no comparable predecessor; start over from here.
            self._tracks = []
            return []

        active = np.abs(current - previous) >= cfg.min_delta
        if active.mean() >= cfg.max_area_fraction:
            # Scene transition. Clear the tracks rather than letting them
            # absorb the whole frame and emit a board-sized "spell".
            self._tracks = []
            return []

        blobs = [b for b in connected_components(active) if b.area >= cfg.min_area]
        return self._advance(blobs, now)

    def _advance(self, blobs, now: float) -> list[SpellEvent]:
        """Match blobs to tracks, then emit whatever peaked."""
        cfg = self.config
        unmatched = list(blobs)
        events: list[SpellEvent] = []

        for track in self._tracks:
            best, best_d = None, cfg.max_drift_px
            for blob in unmatched:
                cx, cy = blob.center
                d = float(np.hypot(cx - track.cx, cy - track.cy))
                if d <= best_d:
                    best, best_d = blob, d
            if best is None:
                track.missed += 1
                # A track that disappears having already peaked is a spell
                # that finished; one that disappears still growing was noise.
                if track.missed > cfg.max_gap_frames and track.shrinking:
                    event = self._emit(track, now)
                    if event is not None:
                        events.append(event)
                continue

            unmatched.remove(best)
            track.missed = 0
            track.last_seen = now
            track.cx, track.cy = best.center
            if best.area > track.peak_area:
                track.peak_area = best.area
                track.peak_cx, track.peak_cy = best.center
                track.peak_extent = max(best.width, best.height) / 2.0
            elif best.area < track.peak_area:
                # First frame of contraction: the VFX is at full extent, and
                # this is the moment the radius measurement is best.
                track.shrinking = True
                event = self._emit(track, now)
                if event is not None:
                    events.append(event)
            track.area = best.area

        for blob in unmatched:
            cx, cy = blob.center
            self._tracks.append(_Track(
                cx=cx, cy=cy, area=blob.area, onset_area=blob.area,
                peak_area=blob.area, first_seen=now, last_seen=now,
                peak_cx=cx, peak_cy=cy,
                peak_extent=max(blob.width, blob.height) / 2.0))

        self._tracks = [t for t in self._tracks
                        if t.missed <= cfg.max_gap_frames
                        and not t.emitted
                        and now - t.first_seen <= cfg.max_duration]
        return events

    # -------------------------------------------------------------- emitting

    def _emit(self, track: _Track, now: float) -> SpellEvent | None:
        """Turn a peaked track into an event, or reject it.

        Marks the track emitted either way: a track that failed the shape
        tests once will keep failing them, and re-testing it every frame
        would emit the moment noise nudged it over a threshold.
        """
        cfg = self.config
        track.emitted = True
        duration = track.last_seen - track.first_seen
        growth = track.peak_area / max(track.onset_area, 1)
        if duration < cfg.min_duration or duration > cfg.max_duration:
            return None
        if growth < cfg.min_growth:
            return None

        projected = self._project(track.peak_cx, track.peak_cy, track.peak_extent)
        if projected is None:
            return None
        tile_x, tile_y, radius = projected

        # Confidence rises with how far the track cleared the two thresholds
        # it is most likely to be a false positive on: it must have grown,
        # and it must have been big. Both are capped so one extreme value
        # cannot carry a weak detection.
        conf = min(1.0, growth / (2.0 * cfg.min_growth)) * \
               min(1.0, track.peak_area / (3.0 * cfg.min_area))
        return SpellEvent(
            tile_x=tile_x, tile_y=tile_y, radius=radius, at=now,
            peak_area=track.peak_area, growth=growth,
            confidence=round(float(conf), 3))

    def _project(self, cx: float, cy: float, extent: float):
        """Downsampled-pixel centre and extent -> arena tiles.

        Radius is measured by projecting the centre and a point one extent to
        its side and taking the tile distance between them, rather than by
        scaling the pixel radius by a constant. The arena is drawn in
        perspective, so a footprint of a given pixel width is a larger number
        of tiles at the far end of the board than at the near end; a constant
        scale would systematically under-read every spell the opponent casts
        on their own half.
        """
        if self.homography is None:
            return None
        step = self.config.downsample
        px, py = cx * step, cy * step
        try:
            tile_x, tile_y = self.homography.pixel_to_tile(px, py)
            edge_x, edge_y = self.homography.pixel_to_tile(px + extent * step, py)
        except ValueError:
            return None
        if self.arena is not None and not (0.0 <= tile_x < self.arena.width
                                           and 0.0 <= tile_y < self.arena.height):
            return None
        radius = float(np.hypot(edge_x - tile_x, edge_y - tile_y))
        return float(tile_x), float(tile_y), radius


# ------------------------------------------------------------- identity


def resolve_identity(
    event: SpellEvent,
    cards: dict,
    candidates=None,
    elixir_drop: float | None = None,
    tolerance: float = 0.9,
) -> str:
    """Name a detected spell, or return "" to leave it unnamed.

    **This is the part that needs no labelled data.** The measured radius on
    its own is a weak signal — several spells land within a tile of each
    other's footprint. It stops being weak once it is narrowed by what the
    opponent can actually be holding, which the deterministic cycle tracker
    already knows (`OpponentTracker.candidate_cards`), and narrowed again by
    what they just paid, which the same tracker's elixir arithmetic gives.
    Radius alone is ambiguous; radius plus cost plus cycle usually is not.

    That is also why this signature takes plain values rather than a tracker:
    the same function names a spell live, and re-labels a recorded frame
    afterwards for `src.live.autolabel`, with no live state involved.

    Returning "" is a real answer. A wrong spell identity feeds a wrong cost
    into the cycle model and corrupts it for the rest of the match, which is
    worse than an unnamed detection that still carries a correct footprint.
    """
    from src.simulator.constants import CardType

    pool = candidates if candidates else list(cards)
    scored = []
    for name in pool:
        stats = cards.get(name)
        if stats is None or stats.type != CardType.SPELL or stats.spell_radius <= 0:
            continue
        if elixir_drop is not None and abs(stats.cost - elixir_drop) > 0.5:
            continue
        error = abs(stats.spell_radius - event.radius)
        if error <= tolerance:
            scored.append((error, name))
    if not scored:
        return ""
    scored.sort()
    if len(scored) > 1 and abs(scored[1][0] - scored[0][0]) < 0.15:
        # Two candidates fit the footprint equally well. Guessing between
        # them is a coin flip that poisons the cycle model when it loses.
        return ""
    return scored[0][1]
