"""The offline harness: perception over recorded frames, no game required.

This is the path a user takes before ever arming the bridge, so the tests
care most about the report being *honest* — a stack that silently found
nothing must say so, and say which knob to turn, rather than printing zeros
that look like a quiet match.
"""
from __future__ import annotations

import json

import pytest
from PIL import Image

from src.live.background import PlateConfig
from src.live.discover import DiscoverConfig
from src.live.homography import Homography
from src.live.replay import (
    ReplayReport,
    format_report,
    frame_paths,
    record,
    replay,
)
from src.simulator.cards import load_cards
from tests.live_frames import render_scene_with_bodies, warmup_frames

PLATE = PlateConfig(warmup_frames=40)
DISCOVER = DiscoverConfig(settle_frames=6, min_area=40)


@pytest.fixture()
def recording(arena, tmp_path):
    """A synthetic 'recording': warmup, then a push from each side."""
    frames = warmup_frames(arena, count=50)
    meta = None
    for i in range(20):
        frame, meta = render_scene_with_bodies(arena, [
            ((6.0, 26.0 - i * 0.7), "hostile"),
            ((12.0, 6.0 + i * 0.7), "friendly"),
        ])
        frames.append(frame)
    for i, frame in enumerate(frames):
        frame.save(tmp_path / f"frame_{i:06d}.png")
    return tmp_path, meta


def _homography(arena, meta):
    return Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})


# ------------------------------------------------------------------ frames


def test_frames_come_back_in_capture_order(tmp_path):
    """Zero padding is why `record` writes frame_000010 rather than frame_10.
    A shuffled match makes every motion-derived answer noise."""
    for name in ("frame_000010.png", "frame_000002.png", "frame_000001.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)

    assert [p.name for p in frame_paths(tmp_path)] == [
        "frame_000001.png", "frame_000002.png", "frame_000010.png"]


def test_non_image_files_are_ignored(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "frame_000001.png")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "report.json").write_text("{}")

    assert len(frame_paths(tmp_path)) == 1


def test_an_empty_directory_reports_nothing_rather_than_crashing(tmp_path):
    report, harvester = replay(tmp_path)

    assert report.frames == 0
    assert harvester is None


# ------------------------------------------------------------------ replay


def test_it_reports_when_the_plate_warmed_up(arena, recording):
    frames_dir, meta = recording

    report, _ = replay(frames_dir, homography=_homography(arena, meta), arena=arena,
                       plate_config=PLATE, discover_config=DISCOVER)

    assert report.plate_ready_at == 39
    assert report.frames == 70
    assert report.size == (556, 1028)


def test_it_finds_both_pushes_and_credits_motion(arena, recording):
    """Teams decided by motion is the number that matters — the fallbacks
    are guesses, and a run dominated by them means tracks are being lost."""
    frames_dir, meta = recording

    report, _ = replay(frames_dir, homography=_homography(arena, meta), arena=arena,
                       plate_config=PLATE, discover_config=DISCOVER)

    assert report.discoveries >= 2
    assert report.by_team["hostile"] >= 1
    assert report.by_team["friendly"] >= 1
    assert report.by_team_source["motion"] == report.discoveries


def test_naming_records_why_it_declined(arena, recording):
    """The skip reasons are the interesting output. "ambiguous between [...]"
    is the system working; a silent zero would not be."""
    frames_dir, meta = recording
    deck = ["knight", "archers", "goblins", "giant",
            "musketeer", "minions", "fireball", "cannon"]

    report, harvester = replay(frames_dir, cards=load_cards(),
                               homography=_homography(arena, meta), arena=arena,
                               deck=deck, plate_config=PLATE,
                               discover_config=DISCOVER)

    assert harvester is not None
    assert report.skipped, "nothing was named and no reason was given"
    assert any("ambiguous" in reason or "no recent play" in reason
               for reason in report.skipped)


def test_sprites_can_be_written_out_for_inspection(arena, recording, tmp_path):
    """A harvest quietly slicing the top off every unit is obvious as a
    picture and nearly invisible as a statistic."""
    frames_dir, meta = recording
    out = tmp_path / "sprites"

    _, harvester = replay(frames_dir, cards=load_cards(),
                          homography=_homography(arena, meta), arena=arena,
                          deck=["cannon"], plate_config=PLATE,
                          discover_config=DISCOVER, save_sprites=out)

    assert out.exists()
    for card in harvester.library.cards:
        written = list((out / card).glob("*.png"))
        assert written, f"{card} banked sprites but wrote none"
        with Image.open(written[0]) as sprite:
            assert sprite.mode == "RGBA", "alpha is the thing most worth seeing"


def test_the_deck_prior_is_the_weakest_the_tracker_could_offer(arena):
    """Replay cannot reconstruct the cycle — that needs play history, not
    pixels — so it models the worst case rather than faking a better one."""
    from src.live.replay import _DeckPrior

    prior = _DeckPrior(["knight", "cannon"])

    assert prior.possible_hand() == []
    assert prior.candidate_cards() == ["knight", "cannon"]
    assert prior.elixir_range == (0.0, 10.0)


def test_the_report_serializes(arena, recording, tmp_path):
    frames_dir, meta = recording

    report, _ = replay(frames_dir, homography=_homography(arena, meta), arena=arena,
                       plate_config=PLATE, discover_config=DISCOVER)

    assert json.loads(json.dumps(report.to_dict()))["frames"] == 70


# ------------------------------------------------------------------ report


def test_a_cold_plate_stops_the_report_rather_than_printing_zeros():
    """Zero discoveries against a plate that never warmed is not a finding
    about the board, and reporting it as one sends you tuning the wrong
    knob."""
    text = format_report(ReplayReport(frames=10, size=(556, 1028)))

    assert "NEVER WARMED UP" in text
    assert "discoveries" not in text


def test_an_empty_board_says_which_knob_to_turn():
    text = format_report(ReplayReport(frames=200, size=(556, 1028),
                                      plate_ready_at=60))

    assert "min_delta" in text and "min_area" in text


def test_lost_tracks_are_called_out(arena):
    """Teams mostly decided by fallbacks means tracks are dying between
    frames, which looks like success until you read the breakdown."""
    from collections import Counter

    text = format_report(ReplayReport(
        frames=200, size=(556, 1028), plate_ready_at=60, discoveries=10,
        by_team_source=Counter({"motion": 2, "side": 8})))

    assert "NOT decided by motion" in text
    assert "max_drift_px" in text


def test_a_slow_capture_is_called_out(arena):
    text = format_report(ReplayReport(frames=50, size=(556, 1028),
                                      capture_interval=0.5,
                                      capture_too_slow=True))

    assert "TOO SLOW" in text


# ------------------------------------------------------------------ record


def test_record_saves_frames_and_taps_nothing(tmp_path):
    """Recording must be safe to run on a live account."""

    class _Device:
        def __init__(self):
            self.taps = []

        def screenshot(self):
            return Image.new("RGB", (556, 1028), (96, 128, 84))

        def tap(self, x, y):
            self.taps.append((x, y))

    device = _Device()

    record(device, tmp_path, frames=5, interval=0.0, log=lambda _: None)

    assert len(frame_paths(tmp_path)) == 5
    assert device.taps == []


def test_recorded_names_sort_in_capture_order(tmp_path):
    class _Device:
        def screenshot(self):
            return Image.new("RGB", (8, 8))

    record(_Device(), tmp_path, frames=12, interval=0.0, log=lambda _: None)

    names = [p.name for p in frame_paths(tmp_path)]
    assert names == sorted(names)
    assert names[0] == "frame_000000.png"
