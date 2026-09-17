"""Model-free discovery: where, what kind, and whose, from geometry alone.

Nothing in this file trains or loads a model, and nothing labels a frame.
That is the point being tested — this is the path that has to work before
any detector exists, or the pipeline never starts without somebody hand
deploying a hundred cards.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.live.background import PlateConfig, RunningPlate
from src.live.discover import TEAM_FROM_MOTION, DiscoverConfig, EntityDiscoverer
from src.live.harvest import FACING_AWAY, FACING_TOWARD
from src.live.vision import TEAM_FRIENDLY, TEAM_HOSTILE
from src.simulator.constants import CardType
from tests.live_frames import render_scene_with_bodies, warmup_frames

PLATE = PlateConfig(warmup_frames=40)
CONFIG = DiscoverConfig(settle_frames=6, min_area=40)


@pytest.fixture()
def plate(arena):
    """A plate built the way the real one is: out of ordinary play."""
    running = RunningPlate(PLATE)
    for frame in warmup_frames(arena, count=60):
        running.update(frame)
    assert running.ready
    return running


def _walk(discoverer, plate, arena, start, step, frames=10, team="hostile"):
    """Feed a unit walking from `start` by `step` tiles each frame."""
    found = []
    for i in range(frames):
        tile = (start[0] + step[0] * i, start[1] + step[1] * i)
        frame, _ = render_scene_with_bodies(arena, [(tile, team)])
        plate.update(frame)
        found.extend(discoverer.observe(frame, now=i * 0.05))
    return found


# ---------------------------------------------------------------- the plate


def test_the_plate_learns_the_empty_arena_from_ordinary_play(arena, plate):
    """No staged empty-arena recording. Every pixel is background most of
    the time, so the running median converges on it for free."""
    from tests.live_frames import render_empty

    empty, _ = render_empty(arena)

    error = np.abs(plate.plate.astype(int) - np.asarray(empty).astype(int)).mean()

    assert error < 1.0


def test_an_empty_frame_shows_no_foreground(arena, plate):
    """If the self-built plate had baked in an afterimage, this is where it
    would show as a permanent phantom unit."""
    from tests.live_frames import render_empty

    assert plate.foreground(render_empty(arena)[0]).sum() == 0


def test_a_unit_shows_as_foreground(arena, plate):
    frame, _ = render_scene_with_bodies(arena, [((9.0, 25.0), "hostile")])

    assert plate.foreground(frame).sum() > 0


def test_nothing_is_reported_before_the_plate_warms_up(arena):
    """A warming plate would cut units out of their own afterimages, so it
    reports "cannot tell yet" rather than a mask full of them."""
    cold = RunningPlate(PlateConfig(warmup_frames=1000))
    discoverer = EntityDiscoverer(cold, config=CONFIG)

    assert _walk(discoverer, cold, arena, (9.0, 25.0), (0.0, 0.6)) == []


def test_a_resized_capture_does_not_corrupt_the_plate(arena, plate):
    from tests.live_frames import render_empty

    small, _ = render_empty(arena, size=(278, 514))

    assert plate.foreground(small) is None


# ------------------------------------------------------------------- kind


def test_a_moving_thing_is_a_troop(arena, plate):
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.8))

    assert found, "a walking unit was never discovered"
    assert found[0].kind == CardType.TROOP


def test_a_stationary_thing_is_a_building(arena, plate):
    """Buildings do not move. That is the entire classifier, and it needs no
    appearance model, no template and no label."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 12.0), (0.0, 0.0))

    assert found
    assert found[0].kind == CardType.BUILDING
    assert found[0].travel_px < CONFIG.static_px


# ------------------------------------------------------------------- team


def test_a_unit_walking_up_the_board_is_ours(arena, plate):
    """You always occupy the bottom seat, so your units advance up the
    screen. That is a fact about the seat, not about the art — no arena skin
    can invalidate it, which is exactly what the hue windows cannot say."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 8.0), (0.0, 0.8))

    assert found
    assert found[0].team == TEAM_FRIENDLY
    assert found[0].team_source == TEAM_FROM_MOTION


def test_a_unit_walking_down_the_board_is_theirs(arena, plate):
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.8))

    assert found
    assert found[0].team == TEAM_HOSTILE
    assert found[0].team_source == TEAM_FROM_MOTION


def test_a_building_falls_back_and_says_so(arena, plate):
    """No motion, no motion signal. The weaker reads are still used, but the
    caller is told which one, so a consumer that cannot afford a wrong team
    can refuse them."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 12.0), (0.0, 0.0))

    assert found
    assert found[0].team_source != TEAM_FROM_MOTION


# ------------------------------------------------------------------ count


def test_a_swarm_reports_its_size(arena, plate):
    """Spawn count is the single most useful thing geometry can offer about
    a deploy -- three bodies at once eliminates most of the roster before
    any appearance is considered."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = []
    for i in range(10):
        # 1.5 tiles apart: far enough that the bodies do not merge into one
        # blob, close enough to still read as a single deploy.
        tiles = [((6.5 + j * 1.5, 26.0 - i * 0.8), "hostile") for j in range(3)]
        frame, _ = render_scene_with_bodies(arena, tiles)
        plate.update(frame)
        found.extend(discoverer.observe(frame, now=i * 0.05))

    assert found
    assert {d.count for d in found} == {3}


# --------------------------------------------------------------- patience


def test_nothing_is_emitted_before_it_has_settled(arena, plate):
    """The decision carried here -- moving or not -- cannot be made from a
    single frame, so emitting early would mean guessing it."""
    discoverer = EntityDiscoverer(plate, config=DiscoverConfig(settle_frames=20,
                                                               min_area=40))

    assert _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.5), frames=8) == []


def test_reset_drops_tracks(arena, plate):
    discoverer = EntityDiscoverer(plate, config=CONFIG)
    _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.5), frames=3)

    discoverer.reset()

    frame, _ = render_scene_with_bodies(arena, [((9.0, 20.0), "hostile")])
    assert discoverer.observe(frame, now=9.0) == []


# ------------------------------------------------------------- handing off


def test_a_discovery_becomes_a_sprite_facing_the_right_way(arena, plate):
    """The one place facing is *observed* rather than assumed: the unit was
    watched walking up or down, which is what decides how it is drawn."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)
    theirs = _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.8))
    assert theirs

    sprite = theirs[0].to_sprite("hog_rider")

    assert sprite.card == "hog_rider"
    assert sprite.team == TEAM_HOSTILE
    assert sprite.facing == FACING_TOWARD
    assert sprite.alpha.any()


def test_our_own_discovery_faces_away(arena, plate):
    discoverer = EntityDiscoverer(plate, config=CONFIG)
    ours = _walk(discoverer, plate, arena, (9.0, 8.0), (0.0, 0.8))
    assert ours

    assert ours[0].to_sprite("knight").facing == FACING_AWAY


def test_a_discovery_is_shaped_like_a_detection(arena, plate):
    """`SelfLabeler` has to consume either without knowing which -- that is
    what lets geometry bootstrap the loop and the detector take over later
    through the same path."""
    from src.live.detector import DetectedEntity

    discoverer = EntityDiscoverer(plate, config=CONFIG)
    found = _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.8))
    assert found

    for attribute in ("card", "kind", "team", "score", "x0", "y0", "x1", "y1", "feet"):
        assert hasattr(found[0], attribute), f"Discovery is missing {attribute}"
    assert set(dir(DetectedEntity)) >= {"feet"}


def test_discovery_never_names_a_card(arena, plate):
    """Geometry answers where, what kind and whose. It has nothing to say
    about which card, and no code path here fills that in."""
    discoverer = EntityDiscoverer(plate, config=CONFIG)

    found = _walk(discoverer, plate, arena, (9.0, 26.0), (0.0, -0.8))

    assert found
    assert all(d.card == "" for d in found)
