"""Labelling real match frames with no human in the loop.

Why synthetic data is not the end of it
---------------------------------------
`synth.py` produces unlimited labelled scenes, and they are wrong in ways
that do not show up until a real frame arrives. Composited pushes scatter
where real ones clump along lanes and bridges. Every harvest artifact -- a
sprite segmented with a bitten-off limb -- is pasted thousands of times.
Nothing is ever partly behind the tower, or lit by a spell going off next
to it, or drawn mid-death-animation. A detector trained only on composites
learns the compositing.

The fix is real frames, and the usual cost of real frames is somebody
labelling them. That cost is avoidable here, because of something this
project already built for an unrelated reason.

The tracker is a labelling oracle
---------------------------------
`OpponentTracker` derives the opponent's elixir and card cycle from observed
play, because strong players do. It happens to be the strongest label source
available, and for free: by the time a card is played, the tracker already
knows the eight-card deck, which four are in hand, and what they can afford.

Stack those constraints against what the detector can see about a new spawn
and the answer is usually forced:

- **kind** -- a new building can only be one of the buildings in hand;
- **spawn count** -- three bodies appearing together is a card with
  ``count == 3``, which cuts most of the roster immediately;
- **affordability** -- the tracker's elixir range rules out what they could
  not have paid for;
- **the hand itself** -- only four cards were available to play at all.

When exactly one card survives all four, that is not a guess, it is a
deduction, and the crop under it is a labelled example on a real frame in
the real skin at the real resolution. When more than one survives, nothing
is banked. A wrong label is worse here than anywhere else in the pipeline:
a wrong identity fed to the tracker corrupts one match, while a wrong label
banked to disk corrupts every model trained after it.

The loop
--------
Play, bank what is deduced, retrain on synthetic plus banked, play again.
The detector improves on exactly the distribution it is deployed against,
and a new arena skin or a new season is a night of self-labelling rather
than an afternoon of anyone's time. Banked frames are written in the same
manifest format `synth.build_dataset` emits, so `train_detector` consumes a
mix of the two without knowing which is which.

Nothing here reads game memory or network traffic; the deductions run on
observed play and on-screen pixels. See CLAUDE.md, "On-Screen Visual
Perception".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.live.synth import Annotation
from src.simulator.constants import CardType


@dataclass(frozen=True)
class AutoLabelConfig:
    """Matching and patience settings for the self-labelling pass."""

    # A detection within this many pixels of one in the previous frame is
    # the same unit moving, not a new spawn.
    max_drift_px: float = 40.0
    # New detections closer together than this are one deploy — which is
    # what makes the spawn count readable, and the count is the constraint
    # that does most of the narrowing.
    group_radius_px: float = 70.0
    # Frames to wait before banking. The spawn frame is deploy VFX, not the
    # unit; banking it would teach the detector that every card looks like
    # a puff of smoke for its first half-second.
    settle_frames: int = 6
    # Give up on a group that never settles — it died on arrival, or the
    # detector lost it.
    max_pending_age: float = 4.0
    # Detections weaker than this do not get banked even when the deduction
    # is unambiguous: the box would be unreliable even if the name is right.
    min_score: float = 0.4


def resolve_spawn(
    kind: CardType,
    count: int,
    cards: dict,
    candidates,
    elixir_max: float | None = None,
) -> tuple[str, str]:
    """Name a newly spawned group, or return ("", reason) when it is not forced.

    Returns the reason either way, because "why was nothing banked" is the
    question anyone tuning this will ask, and a silent empty return makes it
    unanswerable without a debugger.
    """
    if not candidates:
        return "", "no deck prior yet"

    survivors = []
    for name in candidates:
        stats = cards.get(name)
        if stats is None or stats.type != kind:
            continue
        if max(1, stats.count) != count:
            continue
        if elixir_max is not None and stats.cost > elixir_max + 1e-6:
            continue
        survivors.append(name)

    if not survivors:
        return "", f"no {kind.value} in the prior spawns {count}"
    if len(survivors) > 1:
        return "", f"ambiguous between {sorted(survivors)}"
    return survivors[0], "uniquely determined by kind, count and cost"


@dataclass
class _Pending:
    """A resolved spawn waiting for its deploy animation to finish."""

    card: str
    kind: CardType
    team: str
    detections: list
    first_seen: float
    frames: int = 0


@dataclass
class _Banked:
    """One example written to the store, kept for inspection in tests."""

    card: str
    reason: str
    annotations: list


class SelfLabeler:
    """Watches detections across frames and banks what the tracker forces.

    Stateful, like `spells.SpellWatcher` and for the same reason: a spawn is
    only visible as a difference between frames, and the count that does the
    narrowing is only readable at the moment several bodies appear together.
    """

    def __init__(self, cards: dict, config: AutoLabelConfig | None = None, store=None):
        self.cards = cards
        self.config = config or AutoLabelConfig()
        self.store = store
        self._previous: list = []
        self._pending: list[_Pending] = []

    def reset(self) -> None:
        self._previous = []
        self._pending = []

    # ------------------------------------------------------------- stepping

    def observe(self, image, detections, now: float, tracker=None,
                homography=None) -> list[_Banked]:
        """Feed one frame's detections; get back whatever was banked.

        Returns [] on nearly every frame. Banking is supposed to be rare —
        it happens once per opponent deploy at most, and only when the
        deduction is forced.
        """
        cfg = self.config
        usable = [d for d in detections if d.score >= cfg.min_score]
        hostile = [d for d in usable if d.team == "hostile"]

        fresh = [d for d in hostile if not self._matches_previous(d)]
        self._previous = hostile

        # Settle before admitting this frame's groups, not after. A group
        # created on this frame has waited zero frames, and letting it into
        # the settle pass would let `settle_frames=1` bank the deploy
        # animation it exists to wait out.
        banked = self._settle(image, usable, now, homography)

        for group in self._group(fresh):
            pending = self._resolve(group, tracker, now)
            if pending is not None:
                self._pending.append(pending)

        self._pending = [p for p in self._pending
                         if now - p.first_seen <= cfg.max_pending_age]
        return banked

    def _matches_previous(self, detection) -> bool:
        cx, cy = detection.feet
        for other in self._previous:
            ox, oy = other.feet
            if np.hypot(cx - ox, cy - oy) <= self.config.max_drift_px:
                return True
        return False

    def _group(self, fresh) -> list[list]:
        """Cluster simultaneous new detections into single deploys.

        Single-link clustering on feet distance. A swarm lands as one clump,
        so the clump size *is* the card's spawn count — which is why this
        grouping, crude as it is, carries more narrowing power than anything
        else available from one frame.
        """
        groups: list[list] = []
        for detection in fresh:
            cx, cy = detection.feet
            for group in groups:
                if any(np.hypot(cx - o.feet[0], cy - o.feet[1])
                       <= self.config.group_radius_px for o in group):
                    group.append(detection)
                    break
            else:
                groups.append([detection])
        return groups

    def _resolve(self, group, tracker, now: float) -> _Pending | None:
        kinds = {d.kind for d in group}
        if len(kinds) != 1:
            # A clump the detector disagrees about the kind of is not a
            # clean deploy — more likely two things that happened to land
            # near each other. Deducing from it would bank a wrong box.
            return None
        kind = next(iter(kinds))

        candidates, elixir_max = [], None
        if tracker is not None:
            candidates = tracker.possible_hand() or tracker.candidate_cards()
            elixir_max = tracker.elixir_range[1]

        card, _ = resolve_spawn(kind, len(group), self.cards, candidates, elixir_max)
        if not card:
            return None
        return _Pending(card=card, kind=kind, team="hostile",
                        detections=list(group), first_seen=now)

    def _settle(self, image, detections, now: float, homography) -> list[_Banked]:
        """Bank the pending groups whose deploy animation has finished."""
        banked: list[_Banked] = []
        still_pending: list[_Pending] = []

        for pending in self._pending:
            pending.frames += 1
            if pending.frames < self.config.settle_frames:
                still_pending.append(pending)
                continue

            current = [self._nearest(d, detections) for d in pending.detections]
            current = [d for d in current if d is not None]
            if not current:
                continue          # died or lost before it settled

            annotations = [self._annotate(d, pending.card, pending.kind, homography)
                           for d in current]
            if self.store is not None:
                self.store.add(image, annotations)
            banked.append(_Banked(card=pending.card,
                                  reason="settled after deploy animation",
                                  annotations=annotations))

        self._pending = still_pending
        return banked

    def _nearest(self, detection, candidates):
        cx, cy = detection.feet
        best, best_d = None, self.config.max_drift_px * self.config.settle_frames
        for other in candidates:
            ox, oy = other.feet
            d = float(np.hypot(cx - ox, cy - oy))
            if d < best_d:
                best, best_d = other, d
        return best

    def _annotate(self, detection, card: str, kind: CardType, homography) -> Annotation:
        tile_x = tile_y = 0.0
        if homography is not None:
            try:
                tile_x, tile_y = homography.pixel_to_tile(*detection.feet)
            except ValueError:
                pass
        return Annotation(
            card=card, kind=kind.value, team=detection.team,
            x0=int(detection.x0), y0=int(detection.y0),
            x1=int(detection.x1), y1=int(detection.y1),
            tile_x=float(tile_x), tile_y=float(tile_y),
            # Occlusion is not measurable on a real frame the way it is on a
            # composited one — there is no owner map, because nobody placed
            # these. Reporting 1.0 would claim a measurement that was never
            # made, so banked examples carry 0.0 and consumers that filter on
            # visibility should treat real frames separately.
            visibility=0.0)


class LabelStore:
    """Banked real frames, written in `synth`'s manifest format.

    Same format on purpose: `train_detector.SceneDataset` then reads a
    synthetic dataset and a banked one with the same code, and a training
    run mixes them by pointing at both. A separate format would have bought
    nothing and guaranteed the two drifted apart.
    """

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        (self.directory / "images").mkdir(parents=True, exist_ok=True)
        self.scenes: list[dict] = []
        self.cards: set[str] = set()

    def __len__(self) -> int:
        return len(self.scenes)

    def add(self, image, annotations) -> str:
        from PIL import Image

        name = f"{len(self.scenes):06d}.png"
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image)[:, :, :3])
        image.save(self.directory / "images" / name)
        self.scenes.append({"image": f"images/{name}",
                            "annotations": [a.to_dict() for a in annotations]})
        self.cards.update(a.card for a in annotations)
        return name

    def flush(self) -> Path:
        """Write the manifest. Call at the end of a session.

        Written once rather than after every frame: a match banks a handful
        of examples and rewriting the manifest per frame would be the only
        disk cost in the live loop.
        """
        path = self.directory / "manifest.json"
        path.write_text(json.dumps({"cards": sorted(self.cards),
                                    "scenes": self.scenes}))
        return path
