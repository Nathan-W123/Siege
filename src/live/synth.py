"""Synthetic training scenes composited from harvested sprites.

The idea
--------
`harvest.py` gives appearance without context: every sprite was taken alone
on an empty board, so none of them has ever been crowded, occluded, or seen
at the far end of the arena. A detector trained on those directly would be
excellent at the one situation that never occurs in a match.

Compositing closes that gap without anyone labelling anything. Paste the
harvested sprites back onto arena backgrounds at random tiles, in random
combinations, overlapping each other — and because you placed them, you know
every box, class, kind and team exactly. The labels are a *byproduct of
construction*, which is the only kind of label that stays free at scale.

This is the cut-paste-and-learn trick, and it earns its place here for a
specific reason: Clash Royale's arena is a plane viewed by a fixed camera.
There is no novel viewpoint to generalize to, no lighting model to get
wrong. The only geometric transform between "where this sprite was
harvested" and "where it is being pasted" is the perspective scale, and the
homography already knows it exactly.

Where this is weakest, stated plainly
-------------------------------------
Composited scenes get the *appearance* of occlusion right and the
*statistics* of it wrong: real pushes clump along lanes and bridges, while
uniform placement scatters. They also inherit every artifact of the harvest
— a sprite segmented with a bitten-off limb is pasted with that bite
thousands of times. Both are reasons `autolabel.py` exists: real frames,
labelled by the cycle tracker, are what eventually corrects a detector that
learned a compositing artifact. Synthetic data is the bootstrap, not the
destination.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.live.harvest import Sprite, SpriteLibrary
from src.live.vision import DEFAULT_TEAM_COLORS, TEAM_FRIENDLY, TEAM_HOSTILE, rgb_to_hsv
from src.simulator.constants import CardType


@dataclass(frozen=True)
class SynthConfig:
    """Scene composition and domain randomization."""

    min_units: int = 1
    max_units: int = 14
    # Per-sprite scale jitter on top of the perspective scale. Covers card
    # levels (which change nothing visually) far less than it covers client
    # resolution differences and the harvest's own rounding.
    scale_jitter: float = 0.08
    # Appearance randomization. Deliberately wider than any single display
    # varies: the point is a detector that never has to be re-tuned when the
    # arena skin, the day/night variant, or a display colour profile
    # changes, which is exactly what sank the hand-tuned hue windows.
    brightness_jitter: float = 0.18
    contrast_jitter: float = 0.15
    hue_jitter_degrees: float = 8.0
    noise_sigma: float = 4.0
    # An annotation whose sprite ends up less visible than this is dropped.
    # A box around something almost entirely hidden teaches the detector to
    # hallucinate units behind other units.
    min_visibility: float = 0.35
    # Keep sprites off the very edge, where a half-sprite box is mostly
    # background and the crop is unrepresentative.
    margin_tiles: float = 0.5


@dataclass(frozen=True)
class Annotation:
    """One labelled entity in a composited scene.

    The box is the sprite's full pasted extent clipped to the canvas, not
    the extent of its still-visible pixels. A partly occluded unit occupies
    the space it occupies; shrinking the box to the visible sliver would
    teach the detector to under-report anything standing behind anything.
    Heavily occluded sprites are dropped outright instead — see
    `SynthConfig.min_visibility`.
    """

    card: str
    kind: str
    team: str
    x0: int
    y0: int
    x1: int
    y1: int
    tile_x: float
    tile_y: float
    visibility: float

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ colour


def hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Inverse of `vision.rgb_to_hsv`, same conventions and same reasons.

    Hand-rolled to keep `src/live` importable without OpenCV, matching the
    forward conversion it has to round-trip against.
    """
    c = val * sat
    hp = (hue / 60.0) % 6.0
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    z = np.zeros_like(c)
    idx = hp.astype(int) % 6
    r = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [c, x, z, z, x, c])
    g = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [x, c, c, x, z, z])
    b = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5],
                  [z, z, x, c, c, x])
    m = val - c
    return np.clip(np.stack([r + m, g + m, b + m], axis=-1), 0.0, 1.0)


def recolor_team(sprite: Sprite, to_team: str, teams=DEFAULT_TEAM_COLORS) -> Sprite:
    """Retint a sprite's health bar to the other team.

    This is what makes harvesting from your own deploys sufficient. A card's
    body pixels are identical whoever played it — only the bar above it is
    team-tinted — so one harvest pass over your own deck yields hostile
    training examples too, and the opponent never has to cooperate.

    Only pixels inside the source team's hue window are touched, and only
    their hue: saturation and value carry the bar's fill state and its
    shading, and rewriting those would erase the very signal `vision.py`
    reads HP from.
    """
    if sprite.team == to_team:
        return sprite
    source = next((t for t in teams if t.team == sprite.team), None)
    target = next((t for t in teams if t.team == to_team), None)
    if source is None or target is None:
        return sprite

    hue, sat, val = rgb_to_hsv(sprite.rgb)
    tinted = source.mask(hue, sat, val) & sprite.alpha
    if not tinted.any():
        # No bar was captured — a building mid-deploy, or a crop that missed
        # it. Relabelling the team without recolouring anything would be a
        # lie about what the pixels show, so leave it on its own team.
        return sprite

    hue = np.where(tinted, target.hue_center, hue)
    rgb = np.where(tinted[:, :, None],
                   (hsv_to_rgb(hue, sat, val) * 255.0).astype(np.uint8),
                   sprite.rgb)
    return Sprite(card=sprite.card, kind=sprite.kind, team=to_team,
                  rgb=rgb, alpha=sprite.alpha, tile=sprite.tile)


# ---------------------------------------------------------------- geometry


def pixels_per_tile(homography, tile_x: float, tile_y: float) -> float:
    """Screen pixels spanned by one arena tile at this position.

    The arena is drawn in perspective, so this is a real function of
    position and not a constant. It is what tells a sprite harvested near
    the player's own tower how much to shrink when it is pasted at the
    opponent's.
    """
    x0, y0 = homography.tile_to_pixel(tile_x, tile_y)
    x1, y1 = homography.tile_to_pixel(tile_x + 1.0, tile_y)
    return float(np.hypot(x1 - x0, y1 - y0))


def scale_between(homography, source_tile, target_tile) -> float:
    """Resize factor for moving a sprite from one tile to another."""
    source = pixels_per_tile(homography, *source_tile)
    if source <= 1e-6:
        return 1.0
    return pixels_per_tile(homography, *target_tile) / source


def resize_sprite(sprite: Sprite, scale: float) -> Sprite:
    """Scale a sprite, keeping its alpha hard.

    RGB resamples bilinearly and alpha nearest: a bilinear alpha would
    produce fractional edge values that then have to be re-thresholded, and
    the threshold choice would quietly change every box in the dataset.
    """
    if abs(scale - 1.0) < 1e-3:
        return sprite
    h, w = sprite.alpha.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    rgb = np.asarray(Image.fromarray(sprite.rgb).resize(size, Image.BILINEAR))
    alpha = np.asarray(
        Image.fromarray(sprite.alpha.astype(np.uint8) * 255).resize(size, Image.NEAREST))
    return Sprite(card=sprite.card, kind=sprite.kind, team=sprite.team,
                  rgb=rgb, alpha=alpha > 127, tile=sprite.tile)


# ------------------------------------------------------------- compositing


def _paste(canvas: np.ndarray, owner: np.ndarray, sprite: Sprite,
           center: tuple[float, float], index: int):
    """Alpha-composite one sprite and record which pixels it now owns.

    The owner map is how visibility is measured after the whole scene is
    built: a sprite pasted early and buried by three later ones still has
    its box, and only the owner map knows it should not.
    """
    h, w = sprite.alpha.shape
    x0 = int(round(center[0] - w / 2.0))
    y0 = int(round(center[1] - h))          # feet at the anchor, not centre
    ch, cw = canvas.shape[:2]

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(cw, x0 + w), min(ch, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return None

    alpha = sprite.alpha[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    rgb = sprite.rgb[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    window = (slice(dy0, dy1), slice(dx0, dx1))
    canvas[window][alpha] = rgb[alpha]
    owner[window][alpha] = index
    return x0, y0, x0 + w, y0 + h


def jitter_appearance(image: np.ndarray, rng: np.random.Generator,
                      config: SynthConfig) -> np.ndarray:
    """Brightness, contrast, hue and sensor noise, applied to the whole scene.

    Whole-scene rather than per-sprite on purpose: a display colour profile
    or an arena skin shifts everything together, and randomizing sprites
    independently would teach the detector that units and background can
    disagree about their lighting, which they never do.
    """
    out = image.astype(np.float32) / 255.0
    if config.hue_jitter_degrees > 0:
        hue, sat, val = rgb_to_hsv((out * 255).astype(np.uint8))
        shift = rng.uniform(-config.hue_jitter_degrees, config.hue_jitter_degrees)
        out = hsv_to_rgb((hue + shift) % 360.0, sat, val)
    if config.contrast_jitter > 0:
        factor = 1.0 + rng.uniform(-config.contrast_jitter, config.contrast_jitter)
        out = (out - 0.5) * factor + 0.5
    if config.brightness_jitter > 0:
        out = out + rng.uniform(-config.brightness_jitter, config.brightness_jitter)
    if config.noise_sigma > 0:
        out = out + rng.normal(0.0, config.noise_sigma / 255.0, out.shape)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def compose_scene(
    background,
    library: SpriteLibrary,
    homography,
    arena,
    rng: np.random.Generator,
    config: SynthConfig | None = None,
) -> tuple[Image.Image, list[Annotation]]:
    """One labelled training scene.

    Sprites are drawn back-to-front by screen position so nearer units
    occlude farther ones, which is the only occlusion order the game ever
    produces. Drawing in a random order would generate scenes where a unit
    at the far tower is painted over one at the bridge — an image the
    detector will never see and would waste capacity learning.
    """
    config = config or SynthConfig()
    sprites = library.all()
    if not sprites:
        raise ValueError("sprite library is empty; harvest before composing")

    canvas = np.asarray(background)[:, :, :3].copy()
    owner = np.full(canvas.shape[:2], -1, np.int32)

    n = int(rng.integers(config.min_units, config.max_units + 1))
    planned = []
    for _ in range(n):
        sprite = sprites[int(rng.integers(len(sprites)))]
        team = TEAM_HOSTILE if rng.random() < 0.5 else TEAM_FRIENDLY
        tile = (float(rng.uniform(config.margin_tiles, arena.width - config.margin_tiles)),
                float(rng.uniform(config.margin_tiles, arena.height - config.margin_tiles)))
        planned.append((sprite, team, tile))

    # Back to front: larger screen-y is nearer the viewer, so it paints last.
    planned.sort(key=lambda p: homography.tile_to_pixel(*p[2])[1])

    annotations: list[Annotation] = []
    areas: list[int] = []
    for index, (sprite, team, tile) in enumerate(planned):
        placed = recolor_team(sprite, team)
        if sprite.tile is not None:
            scale = scale_between(homography, sprite.tile, tile)
        else:
            scale = 1.0
        scale *= 1.0 + rng.uniform(-config.scale_jitter, config.scale_jitter)
        placed = resize_sprite(placed, max(0.05, scale))

        box = _paste(canvas, owner, placed, homography.tile_to_pixel(*tile), index)
        if box is None:
            annotations.append(None)
            areas.append(0)
            continue
        x0, y0, x1, y1 = box
        annotations.append(Annotation(
            card=placed.card, kind=placed.kind.value, team=team,
            x0=max(0, x0), y0=max(0, y0),
            x1=min(canvas.shape[1], x1), y1=min(canvas.shape[0], y1),
            tile_x=tile[0], tile_y=tile[1], visibility=0.0))
        areas.append(int(placed.alpha.sum()))

    visible = np.bincount(owner[owner >= 0].ravel(), minlength=len(planned))
    kept = []
    for index, annotation in enumerate(annotations):
        if annotation is None or areas[index] == 0:
            continue
        fraction = float(visible[index]) / float(areas[index])
        if fraction < config.min_visibility:
            continue
        kept.append(Annotation(**{**annotation.to_dict(),
                                  "visibility": round(fraction, 3)}))

    return Image.fromarray(jitter_appearance(canvas, rng, config)), kept


def build_dataset(
    out_dir: Path | str,
    background,
    library: SpriteLibrary,
    homography,
    arena,
    scenes: int = 1000,
    seed: int = 0,
    config: SynthConfig | None = None,
) -> Path:
    """Write `scenes` composited images plus one manifest of every box.

    The manifest is a single JSON rather than one sidecar per image: a
    dataset of a hundred thousand scenes is a hundred thousand files of
    filesystem overhead for a few megabytes of boxes, and every consumer
    wants to read the labels without touching the images anyway.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = {"cards": library.cards, "scenes": []}

    for i in range(scenes):
        image, annotations = compose_scene(background, library, homography, arena,
                                           rng, config)
        name = f"{i:06d}.png"
        image.save(out_dir / "images" / name)
        manifest["scenes"].append(
            {"image": f"images/{name}",
             "annotations": [a.to_dict() for a in annotations]})

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path
