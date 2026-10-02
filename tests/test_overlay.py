"""The live overlay: what gets drawn, and the byte order it gets drawn in.

`render` is pure, so all of it is tested here. `Win32Overlay` can only be
put on screen by Windows, but the two conversions most likely to be silently
wrong — premultiplied alpha and BGRA channel order — are static and are
tested here too. Both fail in ways that look like a logic bug rather than a
byte-order one, which is exactly why they are worth pinning.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.live.detector import DetectedEntity
from src.live.overlay import (
    TEAM_COLORS,
    OverlayBox,
    Win32Overlay,
    boxes_for,
    label_for,
    render,
)
from src.simulator.constants import CardType

SIZE = (200, 300)


def _entity(card="knight", kind=CardType.TROOP, team="hostile", score=0.87,
            box=(40, 80, 70, 130)):
    x0, y0, x1, y1 = box
    return DetectedEntity(card=card, kind=kind, team=team, score=score,
                          identity_score=score, x0=x0, y0=y0, x1=x1, y1=y1)


# ------------------------------------------------------------------ labels


def test_a_named_entity_shows_its_name_and_score():
    assert label_for(_entity(card="knight", score=0.87)) == "knight 0.87"


def test_an_unnamed_entity_shows_its_kind_and_a_question_mark():
    """The honesty that matters at a glance.

    Geometry never names anything and its score is 1.0 because it either
    found something or did not. Printing `knight 1.00` there would be wrong
    twice: no card was identified, and there is no identity confidence to
    report. The kind *was* established, so that is what shows.
    """
    assert label_for(_entity(card="", kind=CardType.BUILDING, score=1.0)) == "building ?"


def test_confidence_can_be_suppressed():
    assert label_for(_entity(card="knight"), show_confidence=False) == "knight"


def test_a_discovery_labels_as_cleanly_as_a_detection():
    """`Discovery` and `DetectedEntity` go through the same path, so the
    overlay does not care which stage of the pipeline is running."""
    from src.live.discover import Discovery

    discovery = Discovery(kind=CardType.TROOP, team="friendly",
                          x0=10.0, y0=20.0, x1=30.0, y1=60.0,
                          rgb=np.zeros((40, 20, 3), np.uint8),
                          alpha=np.ones((40, 20), bool))

    assert label_for(discovery) == "troop ?"


# ------------------------------------------------------------------- colour


def test_team_decides_colour_and_nothing_else_does():
    """Blue yours, red theirs — the one property worth reading without
    focusing on it."""
    ours = boxes_for([_entity(team="friendly")])[0]
    theirs = boxes_for([_entity(team="hostile")])[0]

    assert ours.color == TEAM_COLORS["friendly"]
    assert theirs.color == TEAM_COLORS["hostile"]
    assert ours.color != theirs.color


def test_an_unknown_team_is_not_silently_coloured_as_a_friend():
    assert boxes_for([_entity(team="")])[0].color not in TEAM_COLORS.values()


# ------------------------------------------------------------------ drawing


def test_the_layer_is_transparent_where_nothing_was_drawn():
    """It is an overlay: everything not drawn has to stay see-through, or it
    covers the game it is annotating."""
    layer = render(SIZE, boxes_for([_entity()]))

    assert layer.mode == "RGBA"
    assert layer.getpixel((5, 250))[3] == 0


def test_the_box_outline_lands_on_the_box():
    layer = render(SIZE, [OverlayBox(40, 80, 70, 130, (255, 0, 0), "")])
    pixels = np.asarray(layer)

    assert pixels[80, 50, 3] == 255, "top edge missing"
    assert pixels[130, 50, 3] == 255, "bottom edge missing"
    assert pixels[100, 40, 3] == 255, "left edge missing"
    assert pixels[100, 70, 3] == 255, "right edge missing"
    assert pixels[100, 55, 3] == 0, "the box should be hollow, not filled"


def test_the_outline_stays_thin():
    """A thick box hides the unit, and whether the box is on the unit is the
    thing you are usually checking."""
    pixels = np.asarray(render(SIZE, [OverlayBox(40, 80, 70, 130, (255, 0, 0), "")]))

    column = pixels[80:140, 40, 3]
    assert int((pixels[80, 40:71, 3] > 0).sum()) == 31
    assert int((column > 0).sum()) == 51


def test_a_label_is_drawn_on_a_backing_plate():
    """Over grass or a spell going off, bare text is unreadable."""
    plain = np.asarray(render(SIZE, [OverlayBox(40, 80, 70, 130, (255, 0, 0), "")]))
    labelled = np.asarray(render(SIZE, [OverlayBox(40, 80, 70, 130, (255, 0, 0),
                                                   "knight 0.87")]))

    assert (labelled[:, :, 3] > 0).sum() > (plain[:, :, 3] > 0).sum()


def test_a_label_with_no_room_above_stays_inside_the_frame():
    """Units spend much of a match near the edges, and a label that runs off
    the capture is a label you cannot read."""
    layer = render(SIZE, [OverlayBox(2, 0, 40, 30, (255, 0, 0), "mega_knight 0.42")])

    assert layer.size == SIZE
    assert np.asarray(layer)[:, :, 3].any()


def test_a_degenerate_box_is_skipped_not_drawn():
    assert not np.asarray(render(SIZE, [OverlayBox(50, 50, 50, 50, (255, 0, 0),
                                                   "x")]))[:, :, 3].any()


def test_a_box_entirely_off_frame_does_not_crash():
    render(SIZE, [OverlayBox(-80, -80, -10, -10, (255, 0, 0), "x")])


def test_compositing_over_a_frame_keeps_the_frame():
    """What the replay annotations want: boxes on the picture, not alone."""
    frame = Image.new("RGB", SIZE, (96, 128, 84))

    out = render(SIZE, boxes_for([_entity()]), over=frame)

    assert out.size == SIZE
    assert out.getpixel((5, 250))[:3] == (96, 128, 84)


# ------------------------------------------------- the windows conversions


def test_channels_are_reordered_to_bgra():
    """Wrong order swaps red and blue, so the team colours invert — which
    reads as a logic bug in the overlay rather than a byte-order one."""
    image = Image.new("RGBA", (1, 1), (10, 20, 30, 255))

    b, g, r, a = Win32Overlay._premultiplied_bgra(image)[0, 0]

    assert (r, g, b, a) == (10, 20, 30, 255)


def test_alpha_is_premultiplied():
    """`UpdateLayeredWindow` wants each channel already multiplied by its own
    alpha. Hand it straight RGBA and everything blended renders too bright
    with pale fringes, which looks like a colour-space problem and is not."""
    image = Image.new("RGBA", (1, 1), (200, 100, 50, 128))

    b, g, r, a = Win32Overlay._premultiplied_bgra(image)[0, 0]

    assert a == 128
    assert (r, g, b) == (200 * 128 // 255, 100 * 128 // 255, 50 * 128 // 255)


def test_a_fully_transparent_pixel_premultiplies_to_nothing():
    image = Image.new("RGBA", (1, 1), (255, 255, 255, 0))

    assert tuple(Win32Overlay._premultiplied_bgra(image)[0, 0]) == (0, 0, 0, 0)


def test_the_conversion_is_contiguous_for_memmove():
    """It is handed to `ctypes.memmove`, which reads a flat buffer. A view
    with strides would copy garbage into the window."""
    buffer = Win32Overlay._premultiplied_bgra(Image.new("RGBA", (4, 3)))

    assert buffer.flags["C_CONTIGUOUS"]
    assert buffer.shape == (3, 4, 4)


def test_constructing_the_window_off_windows_says_what_to_do_instead():
    """A platform error that names the alternative beats one that does not."""
    import ctypes

    if hasattr(ctypes, "windll"):
        pytest.skip("running on Windows; the window is constructible here")

    with pytest.raises(RuntimeError, match="only available on Windows"):
        Win32Overlay()
