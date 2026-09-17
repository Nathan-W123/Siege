"""Label-free sprite harvesting, against rendered frames only.

Every sprite here is produced the way the real pipeline produces one: play a
card you chose onto an empty board and subtract the empty board. Nothing is
hand-labelled in the tests either, which is the point.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.live.harvest import (
    HarvestConfig,
    Sprite,
    SpriteLibrary,
    build_plate,
    close_mask,
    foreground_mask,
    harvest,
    harvest_sequence,
)
from src.live.vision import TEAM_FRIENDLY, TEAM_HOSTILE
from src.simulator.constants import CardType
from tests.live_frames import render_empty, render_unit_with_body

CONFIG = HarvestConfig(min_area=40)


@pytest.fixture()
def plate(arena):
    image, _ = render_empty(arena)
    return build_plate([image])


# --------------------------------------------------------------- morphology


def test_close_bridges_an_interior_gap():
    """A leg and a weapon separated by background are one sprite."""
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:16] = True
    mask[10:30, 20:26] = True      # 4px of background between the two

    closed = close_mask(mask, radius=3)

    assert closed[20, 18], "the gap between the two parts should be filled"


def test_close_does_not_grow_the_silhouette():
    """The silhouette becomes a bounding box downstream. A dilated one
    teaches the detector boxes that are systematically too big."""
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True

    closed = close_mask(mask, radius=3)

    ys, xs = np.nonzero(closed)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (10, 29, 10, 29)


# -------------------------------------------------------------------- plate


def test_plate_median_rejects_a_transient(arena):
    """A particle effect present in one frame must not enter the plate, or
    it subtracts as a permanent faint sprite on every harvest after."""
    empty, _ = render_empty(arena)
    flash = np.array(empty)
    flash[100:140, 100:140] = (255, 255, 255)
    frames = [empty, empty, Image.fromarray(flash), empty, empty]

    plate = build_plate(frames)

    assert np.array_equal(plate[100:140, 100:140], np.array(empty)[100:140, 100:140])


def test_empty_frame_against_its_own_plate_is_blank(arena, plate):
    """The HUD is in both the plate and the frame, so it must subtract out.
    If it does not, every harvested sprite carries a card-slot chip."""
    empty, _ = render_empty(arena)

    assert not foreground_mask(empty, plate, CONFIG).any()


def test_a_mismatched_frame_size_is_rejected(arena, plate):
    """Silently harvesting against a stale plate would produce garbage
    sprites that look plausible until they poison a training run."""
    small, _ = render_empty(arena, size=(278, 514))

    with pytest.raises(ValueError):
        foreground_mask(small, plate, CONFIG)


# ------------------------------------------------------------------ harvest


def test_harvest_labels_by_construction(arena, plate):
    """The whole claim: the sprite arrives named, and nobody named it."""
    frame, _ = render_unit_with_body(arena, (9.0, 20.0), team=TEAM_HOSTILE)

    sprites = harvest(frame, plate, "cannon", kind=CardType.BUILDING,
                      team=TEAM_HOSTILE, tile=(9.0, 20.0), config=CONFIG)

    assert len(sprites) == 1
    assert sprites[0].card == "cannon"
    assert sprites[0].kind == CardType.BUILDING
    assert sprites[0].tile == (9.0, 20.0)


def test_harvested_alpha_covers_the_body_and_the_bar(arena, plate):
    """Both parts, one sprite. A harvest that split them would paste half a
    unit forever after."""
    frame, _ = render_unit_with_body(arena, (9.0, 20.0))

    sprite = harvest(frame, plate, "knight", config=CONFIG)[0]

    from tests.live_frames import BAR_WIDTH, BODY_RADIUS
    assert sprite.size[0] >= min(BAR_WIDTH, BODY_RADIUS * 2)
    assert sprite.size[1] > BODY_RADIUS          # bar sits above the body
    assert sprite.area > 0


def test_a_swarm_yields_several_bodies(arena, plate):
    """`count` is what keeps three skeletons from harvesting as one blob."""
    from tests.live_frames import render_frame, PlannedUnit, _draw_disc, BODY_RGB
    from tests.live_frames import perspective_camera

    image, _ = render_frame(arena, [])
    frame = np.array(image)
    project = perspective_camera(arena)
    for tile in ((5.0, 20.0), (9.0, 20.0), (13.0, 20.0)):
        px, py = project(*tile)
        _draw_disc(frame, px, py, 11, BODY_RGB)

    sprites = harvest(Image.fromarray(frame), plate, "goblins", count=3, config=CONFIG)

    assert len(sprites) == 3


def test_count_is_a_cap_not_a_quota(arena, plate):
    """Two overlapping skeletons are genuinely one component. Splitting a
    blob down the middle to hit a quota invents a boundary."""
    frame, _ = render_unit_with_body(arena, (9.0, 20.0))

    sprites = harvest(frame, plate, "skeletons", count=3, config=CONFIG)

    assert len(sprites) == 1


def test_speckle_noise_is_filtered_out(arena, plate):
    """Compression noise and plate jitter must not become training sprites."""
    empty, _ = render_empty(arena)
    noisy = np.array(empty)
    noisy[300, 300] = (255, 255, 255)
    noisy[305, 312] = (255, 255, 255)

    assert harvest(Image.fromarray(noisy), plate, "knight", config=CONFIG) == []


def test_a_frame_wide_change_is_refused(arena, plate):
    """A lighting change defeats subtraction entirely, and the failure looks
    exactly like one enormous sprite. Harvesting it would pollute the card
    it was labelled with."""
    empty, _ = render_empty(arena)
    brighter = np.clip(np.asarray(empty).astype(np.int16) + 60, 0, 255).astype(np.uint8)

    assert harvest(Image.fromarray(brighter), plate, "knight", config=CONFIG) == []


def test_an_empty_frame_harvests_nothing(arena, plate):
    """A unit that had already died is a normal thing to hit while stepping
    a recording, not an error."""
    empty, _ = render_empty(arena)

    assert harvest(empty, plate, "knight", config=CONFIG) == []


def test_sequence_skips_the_deploy_vfx(arena, plate):
    """The first half-second after a placement is spawn animation. Harvested
    frames from there contain a cloud, not the card."""
    empty, _ = render_empty(arena)
    unit, _ = render_unit_with_body(arena, (9.0, 20.0))
    frames = [empty, empty, empty, unit, unit]

    sprites = harvest_sequence(frames, plate, "knight", skip_frames=3, config=CONFIG)

    assert len(sprites) == 2, "only the two post-VFX frames should harvest"


# ------------------------------------------------------------------ library


def test_library_round_trips_through_disk(arena, plate, tmp_path):
    """Sprites are the expensive artifact of a harvest session — losing
    their kind or their harvest tile on reload would silently break the
    perspective rescale in `synth`."""
    frame, _ = render_unit_with_body(arena, (9.0, 20.0))
    library = SpriteLibrary()
    library.extend(harvest(frame, plate, "cannon", kind=CardType.BUILDING,
                           team=TEAM_FRIENDLY, tile=(9.0, 20.0), config=CONFIG))
    library.extend(harvest(frame, plate, "knight", tile=(9.0, 20.0), config=CONFIG))

    library.save(tmp_path)
    loaded = SpriteLibrary.load(tmp_path)

    assert loaded.cards == ["cannon", "knight"]
    assert len(loaded) == len(library)
    restored = loaded.get("cannon")[0]
    original = library.get("cannon")[0]
    assert restored.kind == CardType.BUILDING
    assert restored.team == TEAM_FRIENDLY
    assert restored.tile == (9.0, 20.0)
    assert np.array_equal(restored.rgb, original.rgb)
    assert np.array_equal(restored.alpha, original.alpha)


def test_library_load_of_an_empty_directory_is_empty(tmp_path):
    assert len(SpriteLibrary.load(tmp_path)) == 0
