"""Label-free sprite harvesting by background subtraction.

The constraint this removes
---------------------------
`identify.py` names its own cost in its docstring: "our *own* units can be
labelled semi-automatically ... but opponent units need hand labelling."
That sentence is the reason live perception never got past health bars, and
it rests on an assumption worth challenging — that you need the *opponent*
to show you a card in order to learn what it looks like.

You do not. You need *someone* to play it while you are watching, and you
can play it yourself. A card's sprite is the same sprite whoever deploys it;
only the health-bar tint differs, and that is one hue rotation away
(`src.live.synth.recolor_team`).

So: deploy one known card onto an otherwise empty arena, and difference the
frame against a plate of that same empty arena. What is left is that card's
pixels with a pixel-exact alpha, and its name is known because you chose it.
Step through the frames and every animation pose and facing arrives free.
Nothing is hand-labelled. Nothing site-specific is shipped — the sprites
come from your client, your skin, your resolution, which is also why
`identify.py` refuses to bundle templates.

What this does not solve
------------------------
Harvesting gives *appearance*, not *behaviour*. It cannot tell you how a
card moves, and a sprite harvested alone on an empty board has never been
occluded by anything. Both gaps are closed downstream: `synth.py` composites
the harvested sprites into crowded, overlapping scenes, and
`autolabel.py` re-labels real match frames once the detector is good enough
to bootstrap from.

Practical notes that decide whether a harvest is usable
-------------------------------------------------------
**Skip the deploy VFX.** The first half-second after a placement is spawn
animation, not the unit. Frames taken there harvest a cloud. `skip_frames`
on `harvest_sequence` exists for this and defaults high enough to clear it.

**Harvest on both halves of the board.** The arena is drawn in perspective,
so a sprite harvested at the far end is smaller *and* differently
foreshortened. `Sprite.tile` records where it was taken so `synth.py` can
rescale it correctly when it pastes it somewhere else.

**The health bar comes along, and should.** It is part of what the unit
looks like on screen, and a detector trained on bar-less sprites would
mysteriously degrade the moment it saw a real frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.live.vision import TEAM_FRIENDLY, connected_components
from src.simulator.constants import CardType


@dataclass(frozen=True)
class HarvestConfig:
    """Segmentation thresholds for background subtraction."""

    # Per-channel difference from the plate that counts as "not background".
    # Low, on purpose: a missed limb is a bite out of the silhouette, while
    # a few stray background pixels are trimmed by the area filter below.
    min_delta: float = 22.0
    # Components smaller than this are compression noise and plate jitter.
    min_area: int = 60
    # Binary-close radius, in pixels. Bridges the background-coloured gaps
    # that split one unit's silhouette into several components — the gap
    # between a leg and a weapon is background, but the leg and the weapon
    # are one sprite.
    close_radius: int = 3
    # Refuse a component wider or taller than this fraction of the frame.
    # A lighting change or a scrolled background defeats subtraction
    # entirely, and the failure looks like one enormous "sprite".
    max_extent_fraction: float = 0.5


@dataclass(frozen=True)
class Sprite:
    """One harvested appearance of one card.

    `alpha` is a hard boolean mask rather than a soft matte. Antialiased
    edges would be nicer, but background subtraction cannot recover true
    opacity — a half-transparent edge pixel and a fully opaque pixel that
    happens to resemble the background are indistinguishable from two
    frames. Claiming a matte here would be inventing precision.
    """

    card: str
    kind: CardType
    team: str
    rgb: np.ndarray            # (h, w, 3) uint8
    alpha: np.ndarray          # (h, w) bool
    tile: tuple[float, float] | None = None   # where it was harvested

    @property
    def size(self) -> tuple[int, int]:
        return self.alpha.shape[1], self.alpha.shape[0]

    @property
    def area(self) -> int:
        return int(self.alpha.sum())


# --------------------------------------------------------------- morphology


def _shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """`mask` translated by (dy, dx), zero-filled. Not `np.roll`: wrapping
    would let a sprite at one edge of the frame dilate into the other."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = mask[y0:y1, x0:x1]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= _shifted(mask, dy, dx)
    return out


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary close — dilate, then erode by the same radius.

    Fills interior gaps without growing the silhouette, which matters
    because the silhouette becomes a bounding box downstream and a dilated
    one would teach the detector boxes that are systematically too big.
    """
    if radius <= 0:
        return mask
    return ~_dilate(~_dilate(mask, radius), radius)


# ------------------------------------------------------------------- plate


def build_plate(frames) -> np.ndarray:
    """The empty-arena reference, as a per-pixel median over `frames`.

    Median rather than mean: the arena is never perfectly still — ambient
    particle effects, the river animation, a drifting cloud shadow. A mean
    smears every one of those into the plate and they then subtract as
    permanent faint "sprites"; a median discards any of them that is not
    present in most frames.
    """
    stack = np.stack([np.asarray(f)[:, :, :3] for f in frames]).astype(np.float32)
    if stack.shape[0] == 0:
        raise ValueError("need at least one frame to build a plate")
    return np.median(stack, axis=0).astype(np.uint8)


def foreground_mask(frame, plate: np.ndarray,
                    config: HarvestConfig | None = None) -> np.ndarray:
    """Boolean mask of what is on the arena that was not on the plate.

    Compares per channel and takes the maximum rather than comparing luma:
    a sprite can be a different hue at the same brightness as the grass it
    stands on, and a luma comparison drops it entirely.
    """
    config = config or HarvestConfig()
    array = np.asarray(frame)[:, :, :3].astype(np.float32)
    if array.shape != plate.shape:
        raise ValueError(f"frame {array.shape} does not match plate {plate.shape}")
    delta = np.abs(array - plate.astype(np.float32)).max(axis=2)
    return close_mask(delta >= config.min_delta, config.close_radius)


# ----------------------------------------------------------------- harvest


def harvest(
    frame,
    plate: np.ndarray,
    card: str,
    kind: CardType = CardType.TROOP,
    count: int = 1,
    team: str = TEAM_FRIENDLY,
    tile: tuple[float, float] | None = None,
    config: HarvestConfig | None = None,
) -> list[Sprite]:
    """Sprites for `card` in one frame, labelled by construction.

    `count` is the card's spawn count, so a swarm yields its several bodies
    rather than one merged blob. It is a cap, not a requirement: a frame
    where two skeletons overlap legitimately produces fewer components than
    the card spawns, and returning what was actually separable beats
    splitting a blob down the middle to hit a quota.

    Returns [] rather than raising when nothing segments. A frame where the
    unit had already died, or one caught mid-transition, is a normal thing
    to encounter while stepping a recording.
    """
    config = config or HarvestConfig()
    mask = foreground_mask(frame, plate, config)
    array = np.asarray(frame)[:, :, :3]
    height, width = mask.shape
    limit_h = height * config.max_extent_fraction
    limit_w = width * config.max_extent_fraction

    blobs = [b for b in connected_components(mask)
             if b.area >= config.min_area and b.height <= limit_h and b.width <= limit_w]
    blobs.sort(key=lambda b: b.area, reverse=True)

    sprites = []
    for blob in blobs[:max(1, count)]:
        window = (slice(blob.y0, blob.y1 + 1), slice(blob.x0, blob.x1 + 1))
        sprites.append(Sprite(
            card=card, kind=kind, team=team,
            rgb=array[window].copy(), alpha=mask[window].copy(), tile=tile))
    return sprites


def harvest_sequence(
    frames,
    plate: np.ndarray,
    card: str,
    kind: CardType = CardType.TROOP,
    count: int = 1,
    team: str = TEAM_FRIENDLY,
    tile: tuple[float, float] | None = None,
    skip_frames: int = 12,
    config: HarvestConfig | None = None,
) -> list[Sprite]:
    """Harvest every frame of a recorded deploy, skipping the spawn VFX.

    `skip_frames` defaults to about half a second at 20fps, which clears the
    placement animation on most cards. Siege buildings and the slower
    winding troops take materially longer — `CardStats.deploy_time` is the
    number to raise it by for those, and it is already in the card table.
    """
    sprites: list[Sprite] = []
    for frame in list(frames)[skip_frames:]:
        sprites.extend(harvest(frame, plate, card, kind=kind, count=count,
                               team=team, tile=tile, config=config))
    return sprites


# ----------------------------------------------------------------- library


class SpriteLibrary:
    """Harvested sprites on disk, grouped by card.

    Stored as one `.npz` per card rather than a single archive: a harvest
    session adds one card at a time, and rewriting every card's sprites to
    append one card's is both slow and a good way to lose the lot to an
    interrupted write.
    """

    def __init__(self):
        self._sprites: dict[str, list[Sprite]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._sprites.values())

    @property
    def cards(self) -> list[str]:
        return sorted(self._sprites)

    def add(self, sprite: Sprite) -> None:
        self._sprites.setdefault(sprite.card, []).append(sprite)

    def extend(self, sprites) -> None:
        for sprite in sprites:
            self.add(sprite)

    def get(self, card: str) -> list[Sprite]:
        return list(self._sprites.get(card, ()))

    def all(self) -> list[Sprite]:
        return [s for card in self.cards for s in self._sprites[card]]

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for card, sprites in self._sprites.items():
            payload = {}
            for i, sprite in enumerate(sprites):
                payload[f"rgb_{i}"] = sprite.rgb
                payload[f"alpha_{i}"] = sprite.alpha
                payload[f"meta_{i}"] = np.asarray(
                    [sprite.kind.value, sprite.team,
                     "" if sprite.tile is None else f"{sprite.tile[0]},{sprite.tile[1]}"])
            np.savez_compressed(directory / f"{card}.npz", **payload)

    @classmethod
    def load(cls, directory: Path | str) -> "SpriteLibrary":
        library = cls()
        for path in sorted(Path(directory).glob("*.npz")):
            data = np.load(path)
            card = path.stem
            for key in sorted(k for k in data.files if k.startswith("rgb_")):
                i = key.split("_", 1)[1]
                kind, team, tile = (str(v) for v in data[f"meta_{i}"])
                library.add(Sprite(
                    card=card, kind=CardType(kind), team=team,
                    rgb=data[f"rgb_{i}"], alpha=data[f"alpha_{i}"].astype(bool),
                    tile=tuple(float(v) for v in tile.split(",")) if tile else None))
        return library
