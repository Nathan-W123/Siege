"""Composited training scenes: labels as a byproduct of construction.

The assertions that matter here are not "an image was produced" but that
every label is *right by construction* — the class the sprite was harvested
under, a box that actually contains it, and a perspective scale that matches
the homography the live detector will invert.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.live.harvest import HarvestConfig, SpriteLibrary, build_plate, harvest
from src.live.homography import Homography
from src.live.synth import (
    SynthConfig,
    build_dataset,
    compose_scene,
    hsv_to_rgb,
    pixels_per_tile,
    recolor_team,
    resize_sprite,
    scale_between,
)
from src.live.vision import (
    DEFAULT_TEAM_COLORS,
    TEAM_FRIENDLY,
    TEAM_HOSTILE,
    rgb_to_hsv,
)
from src.simulator.constants import CardType
from tests.live_frames import BODY_RGB, render_empty, render_unit_with_body

CONFIG = HarvestConfig(min_area=40)


@pytest.fixture()
def scene(arena):
    """A plate, a homography, and a two-card library harvested from frames."""
    empty, meta = render_empty(arena)
    plate = build_plate([empty])
    homography = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})

    library = SpriteLibrary()
    frame, _ = render_unit_with_body(arena, (9.0, 20.0), team=TEAM_FRIENDLY)
    library.extend(harvest(frame, plate, "knight", team=TEAM_FRIENDLY,
                           tile=(9.0, 20.0), config=CONFIG))
    library.extend(harvest(frame, plate, "cannon", kind=CardType.BUILDING,
                           team=TEAM_FRIENDLY, tile=(9.0, 20.0), config=CONFIG))
    return empty, plate, homography, library


# ------------------------------------------------------------------ colour


def test_hsv_round_trips():
    """`recolor_team` rewrites hue and keeps saturation and value, so the
    conversion has to be an exact inverse or every retint dims the bar."""
    rgb = np.array([[[222, 40, 38], [44, 96, 226], [96, 128, 84], [12, 12, 14]]], np.uint8)

    back = (hsv_to_rgb(*rgb_to_hsv(rgb)) * 255.0).round().astype(np.uint8)

    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_recolor_moves_the_bar_into_the_other_team_window(scene):
    """One harvest pass over your own deck has to yield hostile examples
    too, or the opponent has to cooperate with your training set."""
    _, _, _, library = scene
    friendly = library.get("knight")[0]

    hostile = recolor_team(friendly, TEAM_HOSTILE)

    window = next(t for t in DEFAULT_TEAM_COLORS if t.team == TEAM_HOSTILE)
    hue, sat, val = rgb_to_hsv(hostile.rgb)
    assert window.mask(hue, sat, val).any(), "no pixel landed in the hostile window"
    assert hostile.team == TEAM_HOSTILE


def test_recolor_leaves_the_body_alone(scene):
    """Only the bar is team-tinted. A retint that touched the body would
    train the detector that a card changes colour with its owner."""
    _, _, _, library = scene
    friendly = library.get("knight")[0]
    body = np.all(friendly.rgb == np.array(BODY_RGB, np.uint8), axis=-1)

    hostile = recolor_team(friendly, TEAM_HOSTILE)

    assert body.any(), "fixture drew no body pixels"
    assert np.array_equal(hostile.rgb[body], friendly.rgb[body])


def test_recolor_to_the_same_team_is_a_no_op(scene):
    _, _, _, library = scene
    friendly = library.get("knight")[0]

    assert recolor_team(friendly, TEAM_FRIENDLY) is friendly


# ---------------------------------------------------------------- geometry


def test_a_tile_spans_fewer_pixels_at_the_far_end(scene):
    """The whole reason a sprite has to be rescaled when it is moved."""
    _, _, homography, _ = scene

    near = pixels_per_tile(homography, 9.0, 2.0)
    far = pixels_per_tile(homography, 9.0, 30.0)

    assert far < near


def test_scale_between_shrinks_going_away(scene):
    _, _, homography, _ = scene

    assert scale_between(homography, (9.0, 2.0), (9.0, 30.0)) < 1.0
    assert scale_between(homography, (9.0, 30.0), (9.0, 2.0)) > 1.0
    assert scale_between(homography, (9.0, 12.0), (9.0, 12.0)) == pytest.approx(1.0)


def test_resize_keeps_alpha_hard(scene):
    """A bilinear alpha would need re-thresholding, and the threshold choice
    would quietly move every box in the dataset."""
    _, _, _, library = scene

    resized = resize_sprite(library.get("knight")[0], 1.7)

    assert resized.alpha.dtype == np.bool_
    assert resized.alpha.any()


# ------------------------------------------------------------- composition


def test_every_annotation_names_a_harvested_card(scene):
    """Labels are a byproduct of construction — nothing else can appear."""
    background, _, homography, library = scene
    config = SynthConfig(min_units=6, max_units=6)

    _, annotations = compose_scene(background, library, homography, arena_of(scene),
                                   np.random.default_rng(0), config)

    assert annotations, "composed nothing"
    assert {a.card for a in annotations} <= set(library.cards)
    assert {a.kind for a in annotations} <= {CardType.TROOP.value, CardType.BUILDING.value}


def test_boxes_stay_inside_the_canvas(scene):
    background, _, homography, library = scene
    width, height = background.size

    image, annotations = compose_scene(background, library, homography, arena_of(scene),
                                       np.random.default_rng(1),
                                       SynthConfig(min_units=8, max_units=8))

    assert image.size == (width, height)
    for a in annotations:
        assert 0 <= a.x0 < a.x1 <= width
        assert 0 <= a.y0 < a.y1 <= height


def test_buried_sprites_are_dropped(scene):
    """A box around something almost entirely hidden teaches the detector to
    hallucinate units behind other units."""
    background, _, homography, library = scene
    config = SynthConfig(min_units=30, max_units=30, min_visibility=0.9,
                         margin_tiles=8.0)   # crowd them into a small patch

    _, annotations = compose_scene(background, library, homography, arena_of(scene),
                                   np.random.default_rng(2), config)

    assert len(annotations) < 30
    assert all(a.visibility >= 0.9 for a in annotations)


def test_visibility_is_reported_not_assumed(scene):
    background, _, homography, library = scene

    _, annotations = compose_scene(background, library, homography, arena_of(scene),
                                   np.random.default_rng(3),
                                   SynthConfig(min_units=10, max_units=10))

    assert all(0.0 < a.visibility <= 1.0 for a in annotations)


def test_composition_is_reproducible_from_a_seed(scene):
    """A dataset you cannot regenerate is a dataset you cannot debug."""
    background, _, homography, library = scene
    config = SynthConfig(min_units=5, max_units=5)

    first = compose_scene(background, library, homography, arena_of(scene),
                          np.random.default_rng(7), config)
    second = compose_scene(background, library, homography, arena_of(scene),
                           np.random.default_rng(7), config)

    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    assert [a.to_dict() for a in first[1]] == [a.to_dict() for a in second[1]]


def test_appearance_randomization_actually_varies(scene):
    """Without it the detector re-learns one display's colour profile, which
    is the failure that sank the hand-tuned hue windows."""
    background, _, homography, library = scene
    config = SynthConfig(min_units=1, max_units=1)

    a = compose_scene(background, library, homography, arena_of(scene),
                      np.random.default_rng(11), config)[0]
    b = compose_scene(background, library, homography, arena_of(scene),
                      np.random.default_rng(12), config)[0]

    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_an_empty_library_is_refused(scene):
    """Composing from nothing would write thousands of blank labelled
    scenes and look like a successful dataset build."""
    background, _, homography, _ = scene

    with pytest.raises(ValueError):
        compose_scene(background, SpriteLibrary(), homography, arena_of(scene),
                      np.random.default_rng(0))


# ---------------------------------------------------------------- dataset


def test_dataset_writes_images_and_a_manifest(scene, tmp_path):
    background, _, homography, library = scene

    path = build_dataset(tmp_path, background, library, homography, arena_of(scene),
                         scenes=3, seed=0, config=SynthConfig(min_units=2, max_units=4))

    manifest = json.loads(path.read_text())
    assert manifest["cards"] == library.cards
    assert len(manifest["scenes"]) == 3
    for entry in manifest["scenes"]:
        assert (tmp_path / entry["image"]).exists()
        assert entry["annotations"]


# ---------------------------------------------------------------- helpers


def arena_of(scene):
    """The arena the fixture was rendered against."""
    from src.simulator.cards import load_arena

    return load_arena()
