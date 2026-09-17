"""Finding and classifying new units with no model and no labels at all.

The cold start this solves
--------------------------
The rest of the pipeline has a chicken-and-egg problem. `autolabel.py` turns
real matches into labelled data, but it needs a detector to find the units
first; the detector needs training data; the training data came from a
harvest pass where somebody deploys every card by hand onto an empty board.

That last step is the one nobody wants to do, and it turns out not to be
needed. Everything required to find a new unit and say what *kind* of thing
it is can be read off geometry, with no appearance model whatsoever:

- **Where it is** — it is foreground against the running background plate
  (`background.RunningPlate`), which builds itself out of ordinary play.
- **Troop or building** — buildings do not move. Track the thing for a
  second: if it has travelled, it is a troop; if it has not, it is a
  building. Spells never persist that long and are handled by
  `spells.py` before they get here.
- **Whose it is** — you always occupy the bottom seat, so your units advance
  *up* the screen and the opponent's advance *down*. The sign of the
  vertical displacement is the team, and unlike a hue window it cannot be
  invalidated by an arena skin.

None of those three is a guess about *which card*, and none of them needs a
single labelled example. Naming comes from elsewhere and is already built:
our own plays are named exactly by `runner.HandCycle`, and the opponent's by
`autolabel.resolve_spawn` deducing from kind, spawn count and affordability.

So the loop starts from nothing: play a match, discoveries get named, named
discoveries become harvested sprites with the correct facing, sprites become
a synthetic dataset, the dataset trains a detector, and the detector finds
what geometry alone cannot. No deploy pass, no empty-arena recording, no
labelling.

Deliberately duck-typed
-----------------------
`Discovery` exposes the same attributes `detector.DetectedEntity` does, so
`autolabel.SelfLabeler` consumes either without knowing which. That is the
whole reason the bootstrap composes: before there is a detector, geometry
feeds the labeller; afterwards, the detector does, through the same path.

Stationary buildings keep one gap, stated plainly: with no motion there is
no team signal, so their team falls back to the health-bar hue and then to
which half of the river they sit on. `team_source` records which was used,
and callers that cannot afford a wrong team should require "motion".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.live.harvest import FACING_AWAY, FACING_TOWARD, Sprite, close_mask
from src.live.vision import (
    DEFAULT_TEAM_COLORS,
    TEAM_FRIENDLY,
    TEAM_HOSTILE,
    connected_components,
    rgb_to_hsv,
)
from src.simulator.constants import CardType

TEAM_FROM_MOTION = "motion"
TEAM_FROM_BAR = "bar"
TEAM_FROM_SIDE = "side"


@dataclass(frozen=True)
class DiscoverConfig:
    """Geometry thresholds. No appearance constants anywhere on purpose."""

    min_delta: float = 22.0
    min_area: int = 60
    close_radius: int = 3
    # A blob within this many pixels of one in the previous frame is the
    # same thing moving.
    max_drift_px: float = 45.0
    # New blobs closer together than this are one deploy, which is what
    # makes the spawn count readable.
    group_radius_px: float = 70.0
    # Frames to watch before deciding. Long enough that a troop has visibly
    # travelled, short enough that it is still near where it landed.
    settle_frames: int = 12
    # Pixels of travel below which a thing is considered bolted down. Set
    # above the jitter a segmentation boundary shows frame to frame, or
    # every building reads as a very slow troop.
    static_px: float = 6.0
    # Vertical travel needed before the sign of it is trusted as a team.
    # A troop pushed sideways along the river can drift with almost no
    # vertical component, and guessing from that is worse than falling
    # through to the bar hue.
    min_vertical_px: float = 4.0
    # Give up on a track that never settles.
    max_age: float = 6.0
    # Refuse a blob taller or wider than this fraction of the frame; that is
    # a lighting change or a transition, not a unit.
    max_extent_fraction: float = 0.5


@dataclass(frozen=True)
class Discovery:
    """A newly appeared entity, found without any trained model.

    `card` is always empty and there is no code path that fills it. This
    class answers where, what kind and whose; naming is somebody else's job
    and needs information that is not on the screen.
    """

    kind: CardType
    team: str
    x0: float
    y0: float
    x1: float
    y1: float
    rgb: np.ndarray
    alpha: np.ndarray
    count: int = 1
    team_source: str = TEAM_FROM_MOTION
    travel_px: float = 0.0
    card: str = ""
    # `SelfLabeler` filters detections on `score`. Geometry either found
    # something or did not, so this is 1.0 rather than a fabricated
    # confidence that would let a threshold silently drop real spawns.
    score: float = 1.0
    identity_score: float = 0.0

    @property
    def feet(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, self.y1)

    def to_sprite(self, card: str, kind: CardType | None = None) -> Sprite:
        """Bank this discovery as a named sprite.

        Facing follows the team, and this is the one place it is known for
        certain rather than assumed: the unit was *observed* walking up or
        down the board, which is exactly what decides which way it is drawn.
        """
        team = self.team
        return Sprite(
            card=card, kind=kind or self.kind, team=team,
            rgb=self.rgb, alpha=self.alpha,
            facing=FACING_AWAY if team == TEAM_FRIENDLY else FACING_TOWARD)


@dataclass
class _Track:
    x: float
    y: float
    blob: object
    first_x: float
    first_y: float
    first_seen: float
    group: int
    frames: int = 0
    missed: int = 0


class EntityDiscoverer:
    """Watches the foreground and reports new entities, classified by motion.

    Stateful: a single frame cannot show that something is new, and it
    certainly cannot show whether it moves.
    """

    def __init__(self, plate=None, homography=None, arena=None,
                 config: DiscoverConfig | None = None):
        self.config = config or DiscoverConfig()
        self.plate = plate
        self.homography = homography
        self.arena = arena
        self._tracks: list[_Track] = []
        self._next_group = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_group = 0

    # ------------------------------------------------------------- stepping

    def observe(self, frame, now: float) -> list[Discovery]:
        """Feed one capture; get back entities that have now settled.

        Returns [] on most frames. A discovery is emitted once, well after
        the thing appeared, because the decision it carries — moving or not —
        cannot be made any sooner.
        """
        cfg = self.config
        if self.plate is None:
            return []
        mask = self.plate.foreground(frame, cfg.min_delta)
        if mask is None:
            # The plate is still warming up. Feeding the caller an empty list
            # is right; feeding it afterimages would not be.
            return []
        mask = close_mask(mask, cfg.close_radius)

        height, width = mask.shape
        blobs = [b for b in connected_components(mask)
                 if b.area >= cfg.min_area
                 and b.height <= height * cfg.max_extent_fraction
                 and b.width <= width * cfg.max_extent_fraction]

        emitted = self._advance(frame, mask, blobs, now)
        self._tracks = [t for t in self._tracks
                        if t.missed <= 2 and now - t.first_seen <= cfg.max_age]
        return emitted

    def _advance(self, frame, mask, blobs, now: float) -> list[Discovery]:
        cfg = self.config
        unmatched = list(blobs)
        ready: list[_Track] = []

        for track in self._tracks:
            best, best_d = None, cfg.max_drift_px
            for blob in unmatched:
                cx, cy = blob.center
                d = float(np.hypot(cx - track.x, cy - track.y))
                if d <= best_d:
                    best, best_d = blob, d
            if best is None:
                track.missed += 1
                continue
            unmatched.remove(best)
            track.missed = 0
            track.frames += 1
            track.x, track.y = best.center
            track.blob = best
            if track.frames >= cfg.settle_frames:
                ready.append(track)

        self._admit(unmatched, now)
        if not ready:
            return []

        # Group size is the spawn count, so it has to be counted over the
        # whole group before any member is described.
        sizes: dict[int, int] = {}
        for track in self._tracks:
            sizes[track.group] = sizes.get(track.group, 0) + 1

        self._tracks = [t for t in self._tracks if t not in ready]
        return [self._describe(frame, mask, t, sizes.get(t.group, 1)) for t in ready]

    def _admit(self, blobs, now: float) -> None:
        """Start tracks for blobs that match nothing, clustering the ones
        that appeared together — a swarm is one deploy, and its size is the
        single most useful thing about it."""
        cfg = self.config
        fresh: list[list] = []
        for blob in blobs:
            cx, cy = blob.center
            for group in fresh:
                if any(np.hypot(cx - o.center[0], cy - o.center[1])
                       <= cfg.group_radius_px for o in group):
                    group.append(blob)
                    break
            else:
                fresh.append([blob])

        for group in fresh:
            gid = self._next_group
            self._next_group += 1
            for blob in group:
                cx, cy = blob.center
                self._tracks.append(_Track(x=cx, y=cy, blob=blob, first_x=cx,
                                           first_y=cy, first_seen=now, group=gid))

    # ------------------------------------------------------------ describing

    def _describe(self, frame, mask, track: _Track, count: int) -> Discovery:
        cfg = self.config
        dx = track.x - track.first_x
        dy = track.y - track.first_y
        travel = float(np.hypot(dx, dy))
        kind = CardType.TROOP if travel > cfg.static_px else CardType.BUILDING

        team, source = self._team(frame, track, dy, kind)
        blob = track.blob
        array = np.asarray(frame)[:, :, :3]
        window = (slice(blob.y0, blob.y1 + 1), slice(blob.x0, blob.x1 + 1))
        alpha = mask[window]

        return Discovery(
            kind=kind, team=team, x0=float(blob.x0), y0=float(blob.y0),
            x1=float(blob.x1 + 1), y1=float(blob.y1 + 1),
            rgb=array[window].copy(), alpha=alpha.copy(),
            count=count, team_source=source, travel_px=travel)

    def _team(self, frame, track: _Track, dy: float, kind: CardType):
        """Whose unit this is, by the strongest signal available.

        Motion first: you occupy the bottom seat, so your units advance up
        the screen (falling pixel y) and the opponent's advance down. That
        is a fact about the seat, not about the art, so no skin can break it.

        A building never moves, so it never gets that signal, and the
        fallbacks are weaker on purpose rather than by accident: the
        health-bar hue is the thing this whole rehaul exists to stop relying
        on, and which half of the river it sits on is merely usual rather
        than true. `team_source` is reported so a caller can refuse them.
        """
        cfg = self.config
        if kind == CardType.TROOP and abs(dy) >= cfg.min_vertical_px:
            return (TEAM_FRIENDLY if dy < 0 else TEAM_HOSTILE), TEAM_FROM_MOTION

        by_bar = self._team_from_bar(frame, track)
        if by_bar is not None:
            return by_bar, TEAM_FROM_BAR

        if self.homography is not None and self.arena is not None:
            try:
                _, tile_y = self.homography.pixel_to_tile(track.x, track.y)
            except ValueError:
                return TEAM_HOSTILE, TEAM_FROM_SIDE
            own_half = tile_y < self.arena.height / 2.0
            return (TEAM_FRIENDLY if own_half else TEAM_HOSTILE), TEAM_FROM_SIDE
        return TEAM_HOSTILE, TEAM_FROM_SIDE

    def _team_from_bar(self, frame, track: _Track) -> str | None:
        """Team by health-bar tint, over the blob only.

        Kept as a fallback and not as the primary read. It is the hand-tuned
        path whose constants shift with every skin; motion does not.
        """
        blob = track.blob
        array = np.asarray(frame)[:, :, :3]
        window = (slice(blob.y0, blob.y1 + 1), slice(blob.x0, blob.x1 + 1))
        crop = array[window]
        if crop.size == 0:
            return None
        hue, sat, val = rgb_to_hsv(crop)
        counts = {t.team: int(t.mask(hue, sat, val).sum()) for t in DEFAULT_TEAM_COLORS}
        best = max(counts, key=counts.get)
        if counts[best] == 0:
            return None
        rest = max((v for k, v in counts.items() if k != best), default=0)
        return best if counts[best] > rest else None
