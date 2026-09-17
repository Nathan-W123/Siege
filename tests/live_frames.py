"""Synthetic annotated frames for the live-vision tests (#34).

There is no way to unit-test perception against a live match, and running
against live servers is out of scope for testing, so the fixtures are saved
frames. Real captures are the goal; these synthetic ones exist so the
pipeline has deterministic coverage from day one and so the *fixture format*
is pinned down before anyone hand-labels a screenshot.

Drop real annotated captures into ``tests/fixtures/live/`` as a
``<name>.png`` plus a ``<name>.json`` of the form::

    {"reference_size": [556, 1028],
     "homography_anchors": {"own_king": [278, 830], ...},
     "units": [{"team": "hostile", "tile": [9.0, 20.0], "hp_fraction": 0.5}, ...]}

`test_vision_blobs.py` picks them up automatically and holds them to the same
assertions as the synthetic frame.

Run this module to (re)generate the synthetic fixture on disk::

    python -m tests.live_frames
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "live"
FRAME_SIZE = (556, 1028)  # (width, height)

# Colours chosen to sit inside `vision.DEFAULT_TEAM_COLORS` hue windows.
HOSTILE_RGB = (222, 40, 38)
FRIENDLY_RGB = (44, 96, 226)
TROUGH_RGB = (26, 26, 30)          # unfilled bar remainder: dark, desaturated
ARENA_RGB = (96, 128, 84)          # muted grass; low saturation on purpose
BAR_WIDTH = 22
BAR_HEIGHT = 4


@dataclass(frozen=True)
class PlannedUnit:
    team: str
    tile: tuple[float, float]
    hp_fraction: float


def perspective_camera(arena, size=FRAME_SIZE):
    """Tile -> pixel with a genuine perspective term (far half narrower)."""
    w, h = size

    def project(tx, ty):
        depth = 1.0 + 0.45 * (ty / arena.height)
        px = w / 2 + (tx - arena.width / 2) / depth * (w / arena.width) * 0.9
        py = h - (ty / arena.height) * h * 0.82 / depth - 40.0
        return px, py

    return project


def anchor_pixels(arena, project) -> dict[str, list[int]]:
    from src.live.homography import anchor_tiles

    return {name: [round(v) for v in project(*tile)]
            for name, tile in anchor_tiles(arena).items()}


def _draw_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, rgb) -> None:
    h_img, w_img = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = rgb


def render_frame(arena, units: list[PlannedUnit], size=FRAME_SIZE,
                 bar_to_feet_px: int = 14, with_hud_distractor: bool = True):
    """Render a frame plus the ground truth needed to score it."""
    project = perspective_camera(arena, size)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:, :] = ARENA_RGB

    for unit in units:
        feet_x, feet_y = project(*unit.tile)
        bar_cx = feet_x
        bar_cy = feet_y - bar_to_feet_px
        filled = max(1, int(round(BAR_WIDTH * unit.hp_fraction)))
        left = int(round(bar_cx - BAR_WIDTH / 2))
        top = int(round(bar_cy - BAR_HEIGHT / 2))
        # Trough first, filled portion over it — same z-order the game uses,
        # and what makes the fill fraction recoverable.
        _draw_rect(frame, left, top, BAR_WIDTH, BAR_HEIGHT, TROUGH_RGB)
        rgb = HOSTILE_RGB if unit.team == "hostile" else FRIENDLY_RGB
        _draw_rect(frame, left, top, filled, BAR_HEIGHT, rgb)

    if with_hud_distractor:
        # A saturated red HUD chip below the arena. Anything that maps
        # off-board must be discarded, or the elixir bar becomes a permanent
        # phantom enemy parked on the agent's own side.
        _draw_rect(frame, 40, size[1] - 25, 60, 6, HOSTILE_RGB)

    return Image.fromarray(frame), {
        "reference_size": list(size),
        "homography_anchors": anchor_pixels(arena, project),
        "units": [{"team": u.team, "tile": list(u.tile), "hp_fraction": u.hp_fraction}
                  for u in units],
    }


DEFAULT_UNITS = [
    PlannedUnit("hostile", (4.0, 21.0), 1.0),
    PlannedUnit("hostile", (13.5, 24.0), 0.5),
    PlannedUnit("hostile", (9.0, 18.0), 0.25),
    PlannedUnit("friendly", (5.0, 9.0), 1.0),
    PlannedUnit("friendly", (14.0, 11.0), 0.75),
]


def write_default_fixture(arena, directory: Path = FIXTURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image, meta = render_frame(arena, DEFAULT_UNITS)
    meta["synthetic"] = True
    image.save(directory / "synthetic_push.png")
    (directory / "synthetic_push.json").write_text(json.dumps(meta, indent=2))
    return directory / "synthetic_push.png"


if __name__ == "__main__":  # pragma: no cover - fixture regeneration
    from src.simulator.cards import load_arena

    print(write_default_fixture(load_arena()))


# ------------------------------------------------------------ spell frames

SPELL_RGB = (255, 196, 64)         # bright VFX bloom; nothing in the arena
                                   # palette is this bright or this warm
# Frame-to-frame brightness modulation of the bloom. Real spell VFX animates
# internally -- particles, a swirling core, a pulsing rim -- so consecutive
# frames differ across the *whole* footprint. A flat disc would differ only
# on the ring it grew by, which understates the footprint by the square of
# the growth rate and would let a detector pass these tests while measuring
# something much smaller than a real spell presents.
SPELL_PULSE = (1.0, 0.8)


def _draw_disc(frame: np.ndarray, cx: float, cy: float, radius: float, rgb) -> None:
    """Filled disc. Spell VFX is round; drawing it as a rectangle would let
    a detector pass the area tests without ever facing a real footprint."""
    h_img, w_img = frame.shape[:2]
    if radius <= 0:
        return
    y0, y1 = max(0, int(cy - radius)), min(h_img, int(cy + radius) + 1)
    x0, x1 = max(0, int(cx - radius)), min(w_img, int(cx + radius) + 1)
    if y1 <= y0 or x1 <= x0:
        return
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    inside = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    frame[y0:y1, x0:x1][inside] = rgb


def bloom_radii(peak: float, grow_frames: int, fade_frames: int) -> list[float]:
    """Radius per frame: a bloom that grows, peaks, then collapses.

    Starts and ends at zero so a detector sees a real onset and a real
    disappearance rather than beginning mid-bloom.
    """
    return ([0.0]
            + [peak * (i + 1) / grow_frames for i in range(grow_frames)]
            + [peak * (fade_frames - i) / (fade_frames + 1) for i in range(fade_frames)]
            + [0.0])


def render_bloom_at_pixel(
    arena,
    center_px: tuple[float, float],
    peak_radius_px: float = 34.0,
    grow_frames: int = 4,
    fade_frames: int = 3,
    size=FRAME_SIZE,
    units: list[PlannedUnit] | None = None,
):
    """Frames of a spell bloom centred on a screen pixel.

    Takes a pixel rather than a tile so the same helper renders a bloom on
    the board and one over the HUD, which is the difference the arena bound
    is supposed to catch.
    """
    cx, cy = center_px
    frames = []
    meta = None
    for i, radius in enumerate(bloom_radii(peak_radius_px, grow_frames, fade_frames)):
        image, meta = render_frame(arena, units or [], size=size)
        frame = np.array(image)
        pulse = SPELL_PULSE[i % len(SPELL_PULSE)]
        _draw_disc(frame, cx, cy, radius,
                   tuple(int(c * pulse) for c in SPELL_RGB))
        frames.append(Image.fromarray(frame))
    return frames, meta


def render_spell_sequence(
    arena,
    tile: tuple[float, float],
    peak_radius_px: float = 34.0,
    grow_frames: int = 4,
    fade_frames: int = 3,
    size=FRAME_SIZE,
    units: list[PlannedUnit] | None = None,
):
    """A spell bloom centred on an arena tile."""
    project = perspective_camera(arena, size)
    return render_bloom_at_pixel(
        arena, project(*tile), peak_radius_px=peak_radius_px,
        grow_frames=grow_frames, fade_frames=fade_frames, size=size, units=units)


def render_march_sequence(
    arena,
    start_tile: tuple[float, float],
    steps: int = 8,
    dy: float = -0.6,
    size=FRAME_SIZE,
):
    """A hostile unit walking down the board, frame by frame.

    The negative case the spell detector has to survive: this produces
    change on every frame, at a real position, for longer than any spell
    lasts. Anything that fires on it would fire on every push.
    """
    frames = []
    meta = None
    for i in range(steps):
        tile = (start_tile[0], start_tile[1] + dy * i)
        image, meta = render_frame(arena, [PlannedUnit("hostile", tile, 1.0)], size=size)
        frames.append(image)
    return frames, meta


# ------------------------------------------------------- units with bodies

BODY_RGB = (170, 140, 110)         # desaturated, so it sits OUTSIDE both team
                                   # hue windows -- a recolour that touched it
                                   # would show up as a failure, not as noise
BODY_RADIUS = 11


def render_unit_with_body(
    arena,
    tile: tuple[float, float],
    team: str = "hostile",
    hp_fraction: float = 1.0,
    size=FRAME_SIZE,
    bar_to_feet_px: int = 14,
    with_hud_distractor: bool = True,
):
    """A unit drawn as a body plus its health bar.

    `render_frame` draws bars alone, which is all the blob detector needs.
    Harvesting needs something bar-shaped *and* something body-shaped: the
    bar is what gets retinted between teams and the body is what must
    survive that retint untouched.
    """
    project = perspective_camera(arena, size)
    image, meta = render_frame(arena, [PlannedUnit(team, tile, hp_fraction)],
                               size=size, bar_to_feet_px=bar_to_feet_px,
                               with_hud_distractor=with_hud_distractor)
    feet_x, feet_y = project(*tile)
    frame = np.array(image)
    _draw_disc(frame, feet_x, feet_y - BODY_RADIUS, BODY_RADIUS, BODY_RGB)
    return Image.fromarray(frame), meta


def render_empty(arena, size=FRAME_SIZE, with_hud_distractor: bool = True):
    """An empty arena frame — the plate `harvest` subtracts against."""
    image, meta = render_frame(arena, [], size=size,
                               with_hud_distractor=with_hud_distractor)
    return image, meta


def render_scene_with_bodies(arena, placements, size=FRAME_SIZE,
                             with_hud_distractor: bool = True):
    """Several units, each drawn as a body plus its health bar.

    `placements` is [(tile, team), ...]. Used for the model-free discovery
    tests, which need a stream of frames where units move and buildings do
    not -- geometry is the only signal there, so the fixture has to get the
    geometry right even though the art is a disc.
    """
    project = perspective_camera(arena, size)
    image, meta = render_frame(
        arena, [PlannedUnit(team, tile, 1.0) for tile, team in placements],
        size=size, with_hud_distractor=with_hud_distractor)
    frame = np.array(image)
    for tile, _ in placements:
        feet_x, feet_y = project(*tile)
        _draw_disc(frame, feet_x, feet_y - BODY_RADIUS, BODY_RADIUS, BODY_RGB)
    return Image.fromarray(frame), meta


def warmup_frames(arena, count=80, size=FRAME_SIZE):
    """A stretch of ordinary play: one unit wandering, never parked.

    This is what a `RunningPlate` is fed in real use -- nobody records an
    empty arena, they just play -- so the tests build their plate the same
    way rather than from a staged empty capture.
    """
    frames = []
    for i in range(count):
        tile = (3.0 + (i % 13), 18.0 + (i % 7))
        frames.append(render_scene_with_bodies(arena, [(tile, "hostile")], size=size)[0])
    return frames
