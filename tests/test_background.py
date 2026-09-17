"""The background plate that builds itself out of ordinary play.

Everything here feeds the estimator the way live play would — a stream of
frames with units in them — rather than a staged empty capture. Removing
that staging is the whole reason this module exists.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.live.background import PlateConfig, RunningPlate
from tests.live_frames import render_empty, render_scene_with_bodies, warmup_frames

CONFIG = PlateConfig(warmup_frames=40)


@pytest.fixture()
def plate(arena):
    running = RunningPlate(CONFIG)
    for frame in warmup_frames(arena, count=60):
        running.update(frame)
    return running


def test_it_converges_on_the_empty_arena_nobody_recorded(arena, plate):
    """The claim that removes a setup step: every pixel of the playfield is
    background most of the time, so the running median is the empty arena
    whether or not anyone staged one."""
    empty = np.asarray(render_empty(arena)[0]).astype(int)

    assert np.abs(plate.plate.astype(int) - empty).mean() < 1.0


def test_the_seed_frame_leaves_no_ghost(arena, plate):
    """The plate is seeded from the first frame, so whatever stood there is
    seeded with it. A ghost that outlives warmup becomes permanent — it
    reads as foreground, and the steady-state rule refuses to learn from
    foreground — and `discover` would faithfully report it as a building
    that never moves."""
    assert plate.foreground(render_empty(arena)[0]).sum() == 0


def test_nothing_is_offered_before_warmup(arena):
    """An honest "cannot tell yet" beats a mask full of afterimages that
    looks exactly like a busy board."""
    cold = RunningPlate(PlateConfig(warmup_frames=1000))
    for frame in warmup_frames(arena, count=20):
        cold.update(frame)

    assert not cold.ready
    assert cold.foreground(render_empty(arena)[0]) is None


def test_a_unit_that_stops_moving_is_not_absorbed(arena, plate):
    """The failure this guards is silent and total: a building parks on a
    tile, the estimator learns it as scenery, and it stops being detectable
    at exactly the moment it starts mattering."""
    still, _ = render_scene_with_bodies(arena, [((9.0, 12.0), "hostile")])
    before = int(plate.foreground(still).sum())
    assert before > 0

    for _ in range(40):
        plate.update(still)

    assert plate.foreground(still).sum() > before * 0.5


def test_something_permanent_is_absorbed_eventually(arena, plate):
    """...but "refuses to learn" cannot mean "never", or a tower that falls
    stays on the plate for the rest of the match."""
    still, _ = render_scene_with_bodies(arena, [((9.0, 12.0), "hostile")])
    before = int(plate.foreground(still).sum())

    for _ in range(1200):
        plate.update(still)

    assert plate.foreground(still).sum() < before * 0.5


def test_background_that_never_settles_is_excluded(arena):
    """The river animates continuously. Without the volatility map it reads
    as a permanent wall of foreground and ruins every harvest near it."""
    empty, _ = render_empty(arena)
    base = np.asarray(empty)
    running = RunningPlate(CONFIG)

    rng = np.random.default_rng(0)
    for _ in range(120):
        frame = base.copy()
        # A band that shimmers every frame — background, but never still.
        frame[500:540, :] = np.clip(
            rng.normal(120, 40, (40, base.shape[1], 3)), 0, 255).astype(np.uint8)
        running.update(Image.fromarray(frame))

    restless = running.unstable()
    assert restless[500:540, :].mean() > 0.8, "the animated band was not flagged"
    assert restless[600:640, :].mean() < 0.2, "static arena was flagged restless"


def test_a_restless_region_is_kept_out_of_the_foreground(arena):
    empty, _ = render_empty(arena)
    base = np.asarray(empty)
    running = RunningPlate(CONFIG)
    rng = np.random.default_rng(0)
    for _ in range(120):
        frame = base.copy()
        frame[500:540, :] = np.clip(
            rng.normal(120, 40, (40, base.shape[1], 3)), 0, 255).astype(np.uint8)
        running.update(Image.fromarray(frame))

    probe = base.copy()
    probe[500:540, :] = 40

    assert running.foreground(Image.fromarray(probe))[500:540, :].sum() == 0


def test_a_resized_capture_is_refused_not_misread(arena, plate):
    """Differencing against a stale plate produces garbage that looks
    plausible right up until it poisons a training set."""
    small, _ = render_empty(arena, size=(278, 514))

    assert plate.foreground(small) is None


def test_reset_starts_over(arena, plate):
    plate.reset()

    assert not plate.ready
    assert plate.plate is None
    assert plate.frames == 0


def test_a_changed_capture_size_reseeds_rather_than_crashing(arena, plate):
    """The window was resized mid-session."""
    small, _ = render_empty(arena, size=(278, 514))

    plate.update(small)

    assert plate.plate.shape[:2] == (514, 278)
    assert plate.frames == 1
