"""Temporal spell detection, against rendered sequences only.

Nothing here touches a live match. The sequences come from
`tests.live_frames`, which renders a bloom that grows and collapses the way
real VFX does, plus the negative case that matters most: a unit walking.
"""
from __future__ import annotations

import pytest

from src.live.homography import Homography
from src.live.spells import SpellWatcher, resolve_identity
from src.simulator.cards import load_cards
from tests.live_frames import render_march_sequence, render_spell_sequence


def _homography(arena, meta):
    return Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})


def _run(watcher, frames, dt: float = 0.05):
    """Feed a sequence and collect everything it emitted."""
    events = []
    for i, frame in enumerate(frames):
        events.extend(watcher.observe(frame, now=i * dt))
    return events


# ------------------------------------------------------------ the positive


def test_bloom_is_detected_at_the_right_tile(arena):
    frames, meta = render_spell_sequence(arena, tile=(9.0, 22.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)

    events = _run(watcher, frames)

    assert len(events) == 1, f"expected exactly one spell, got {len(events)}"
    event = events[0]
    assert event.tile_x == pytest.approx(9.0, abs=1.2)
    assert event.tile_y == pytest.approx(22.0, abs=1.2)
    assert event.radius > 0.5
    assert event.confidence > 0.0


def test_radius_tracks_the_footprint(arena):
    """A bigger bloom must read as a bigger radius.

    This is the property `resolve_identity` rests on: if radius does not
    order spells by their real footprint, narrowing by it is worthless.
    """
    small, meta = render_spell_sequence(arena, (9.0, 16.0), peak_radius_px=20.0)
    large, _ = render_spell_sequence(arena, (9.0, 16.0), peak_radius_px=44.0)

    homography = _homography(arena, meta)
    r_small = _run(SpellWatcher(homography, arena), small)[0].radius
    r_large = _run(SpellWatcher(homography, arena), large)[0].radius

    assert r_large > r_small * 1.5


def test_perspective_is_applied_to_radius(arena):
    """The same pixel footprint is more tiles away than near.

    A constant pixels-per-tile scale would report these as equal and
    systematically under-read every spell cast on the far half.
    """
    near, meta = render_spell_sequence(arena, (9.0, 4.0), peak_radius_px=30.0)
    far, _ = render_spell_sequence(arena, (9.0, 26.0), peak_radius_px=30.0)

    homography = _homography(arena, meta)
    r_near = _run(SpellWatcher(homography, arena), near)[0].radius
    r_far = _run(SpellWatcher(homography, arena), far)[0].radius

    assert r_far > r_near


# ------------------------------------------------------------ the negatives


def test_a_marching_unit_is_not_a_spell(arena):
    """The failure that would make this module useless.

    A push produces change on every frame for longer than any spell lasts.
    Firing on it would put a phantom spell under every troop that moves.
    """
    frames, meta = render_march_sequence(arena, (9.0, 24.0), steps=10)
    watcher = SpellWatcher(_homography(arena, meta), arena)

    assert _run(watcher, frames) == []


def test_scene_transition_emits_nothing(arena):
    """A screen wipe changes nearly every pixel at once.

    Treating that as a spell would fabricate one at the exact moment the
    board is least readable — match start, or a crown-tower cutscene.
    """
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)
    watcher.observe(frames[0], now=0.0)

    import numpy as np
    from PIL import Image
    wipe = Image.fromarray(np.full((frames[0].height, frames[0].width, 3), 250, np.uint8))

    assert watcher.observe(wipe, now=0.05) == []


def test_off_arena_bloom_is_dropped(arena):
    """VFX over the HUD is not a spell on the board.

    Card-slot and elixir-bar animations bloom too; without the arena bound
    they would land as spells somewhere on the playfield.
    """
    frames, meta = render_spell_sequence(arena, (9.0, 20.0), size=(556, 1028))
    homography = _homography(arena, meta)

    from tests.live_frames import render_bloom_at_pixel

    # Same bloom the positive test detects, drawn below the board where the
    # card slots sit -- so this fails only on the arena bound, not because
    # the bloom was too small to see.
    hud, _ = render_bloom_at_pixel(arena, (60.0, 1014.0))

    assert _run(SpellWatcher(homography, arena), hud) == []


def test_first_frame_never_emits(arena):
    """There is no predecessor to difference against."""
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)
    assert watcher.observe(frames[2], now=0.0) == []


def test_reset_clears_carryover(arena):
    """A track surviving a match boundary would emit against a new board."""
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)
    for i, frame in enumerate(frames[:3]):
        watcher.observe(frame, now=i * 0.05)

    watcher.reset()

    assert watcher.observe(frames[4], now=1.0) == []


def test_resized_capture_restarts_cleanly(arena):
    """The window was resized mid-match; frames stop being comparable."""
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)
    watcher.observe(frames[0], now=0.0)

    assert watcher.observe(frames[1].resize((278, 514)), now=0.05) == []


# ------------------------------------------------------------ identity


@pytest.fixture()
def cards():
    return load_cards()


def test_identity_narrowed_by_the_deck_prior(cards, arena):
    """The whole no-labels claim in one test.

    The same measured footprint is ambiguous against the full spell roster
    and unambiguous against what the opponent can still be holding — and the
    cycle tracker supplies that set for free, with nothing hand-labelled.
    """
    from src.live.spells import SpellEvent

    fireball = cards["fireball"]
    event = SpellEvent(tile_x=9.0, tile_y=8.0, radius=fireball.spell_radius,
                       at=1.0, peak_area=200, growth=2.0, confidence=0.8)

    narrowed = resolve_identity(event, cards, candidates=["fireball", "graveyard"])

    assert narrowed == "fireball"


def test_ambiguous_footprint_abstains(cards):
    """Two spells fitting equally well must yield "", not a coin flip.

    A wrong identity feeds a wrong cost into the cycle model and corrupts it
    for the rest of the match; an unnamed spell still carries its footprint.
    """
    from src.live.spells import SpellEvent
    from src.simulator.constants import CardType

    spells = sorted((n for n, c in cards.items()
                     if c.type == CardType.SPELL and c.spell_radius > 0),
                    key=lambda n: cards[n].spell_radius)
    pair = next(((a, b) for a, b in zip(spells, spells[1:])
                 if abs(cards[a].spell_radius - cards[b].spell_radius) < 0.05), None)
    if pair is None:
        pytest.skip("no two spells share a footprint in this card table")

    event = SpellEvent(tile_x=9.0, tile_y=8.0, radius=cards[pair[0]].spell_radius,
                       at=1.0, peak_area=200, growth=2.0, confidence=0.8)

    assert resolve_identity(event, cards, candidates=list(pair)) == ""


def test_elixir_drop_breaks_a_tie(cards):
    """Cost is the second free signal: the tracker already derives it."""
    from src.live.spells import SpellEvent
    from src.simulator.constants import CardType

    spells = [n for n, c in cards.items() if c.type == CardType.SPELL and c.spell_radius > 0]
    pair = next(((a, b) for a in spells for b in spells
                 if a != b
                 and abs(cards[a].spell_radius - cards[b].spell_radius) < 0.05
                 and cards[a].cost != cards[b].cost), None)
    if pair is None:
        pytest.skip("no same-footprint different-cost spell pair in this card table")

    a, b = pair
    event = SpellEvent(tile_x=9.0, tile_y=8.0, radius=cards[a].spell_radius,
                       at=1.0, peak_area=200, growth=2.0, confidence=0.8)

    assert resolve_identity(event, cards, candidates=[a, b]) == ""
    assert resolve_identity(event, cards, candidates=[a, b],
                            elixir_drop=cards[a].cost) == a


def test_unknown_footprint_abstains(cards):
    """A radius matching nothing is not forced onto the nearest spell."""
    from src.live.spells import SpellEvent

    event = SpellEvent(tile_x=9.0, tile_y=8.0, radius=99.0,
                       at=1.0, peak_area=200, growth=2.0, confidence=0.8)

    assert resolve_identity(event, cards) == ""


# ---------------------------------------------------------- capture rate


def test_a_slow_capture_rate_is_reported_not_silent(arena):
    """Being fed slowly does not degrade detection, it stops it.

    A bloom lasts well under a second, so a sampler managing two frames in
    that window sees one flash with no growth to measure. The resulting
    silence is indistinguishable from a match where nobody cast anything,
    which is exactly the kind of failure this project keeps insisting be
    made visible.
    """
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)

    for i in range(8):
        watcher.observe(frames[i % len(frames)], now=i * 1.2)   # the default cooldown

    assert watcher.starved
    assert watcher.frame_interval == pytest.approx(1.2, abs=0.01)


def test_a_healthy_capture_rate_is_not_flagged(arena):
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)

    for i, frame in enumerate(frames):
        watcher.observe(frame, now=i * 0.05)

    assert not watcher.starved


def test_starvation_needs_a_few_frames_before_it_is_claimed(arena):
    """One slow frame is a hiccup, not a capture rate."""
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)

    watcher.observe(frames[0], now=0.0)
    watcher.observe(frames[1], now=2.0)

    assert not watcher.starved


def test_a_match_boundary_gap_is_not_counted_as_the_rate(arena):
    """`reset` happens between matches, and the wait for the next one would
    otherwise read as a permanently starved capture loop."""
    frames, meta = render_spell_sequence(arena, (9.0, 20.0))
    watcher = SpellWatcher(_homography(arena, meta), arena)
    for i, frame in enumerate(frames):
        watcher.observe(frame, now=i * 0.05)

    watcher.reset()
    watcher.observe(frames[0], now=600.0)          # ten minutes between matches

    assert watcher.frame_interval == pytest.approx(0.05, abs=0.01)
