"""Self-labelling: deductions the tracker forces, and nothing else.

The tests split into two halves on purpose. `resolve_spawn` is pure and gets
exhaustive treatment, because it is where a wrong label would come from.
`SelfLabeler` is the stateful shell around it, and is tested for the things
state gets wrong: calling a moving unit a new spawn, banking the deploy
animation, and banking something that died before it settled.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.live.autolabel import (
    AutoLabelConfig,
    LabelStore,
    SelfLabeler,
    resolve_spawn,
)
from src.live.detector import DetectedEntity
from src.simulator.cards import load_cards
from src.simulator.constants import CardType

CONFIG = AutoLabelConfig(settle_frames=2)


@pytest.fixture()
def cards():
    return load_cards()


@pytest.fixture()
def frame():
    return Image.fromarray(np.zeros((64, 64, 3), np.uint8))


def _detection(x=100.0, y=200.0, kind=CardType.TROOP, team="hostile",
               card="", score=0.9):
    return DetectedEntity(card=card, kind=kind, team=team, score=score,
                          identity_score=score,
                          x0=x - 10, y0=y - 30, x1=x + 10, y1=y)


class _Tracker:
    """The slice of `OpponentTracker` the labeller actually consumes."""

    def __init__(self, hand, elixir_max=10.0, deck=None):
        self._hand = list(hand)
        self._deck = list(deck or hand)
        self.elixir_range = (0.0, elixir_max)

    def possible_hand(self):
        return list(self._hand)

    def candidate_cards(self):
        return list(self._deck)


# --------------------------------------------------------------- deductions


def test_kind_and_count_force_a_single_card(cards):
    """The claim: four constraints the project already computes, stacked,
    usually leave exactly one card."""
    card, reason = resolve_spawn(CardType.BUILDING, 1, cards,
                                 ["cannon", "knight", "archers", "fireball"])

    assert card == "cannon"
    assert "uniquely" in reason


def test_spawn_count_separates_two_troops(cards):
    """Three bodies at once is a card with count == 3, which cuts most of
    the roster in one step."""
    triple = next(n for n, c in cards.items()
                  if c.type == CardType.TROOP and c.count == 3)
    single = next(n for n, c in cards.items()
                  if c.type == CardType.TROOP and c.count == 1)

    assert resolve_spawn(CardType.TROOP, 3, cards, [triple, single])[0] == triple
    assert resolve_spawn(CardType.TROOP, 1, cards, [triple, single])[0] == single


def test_affordability_rules_out_what_they_could_not_pay_for(cards):
    cheap = next(n for n, c in cards.items()
                 if c.type == CardType.TROOP and c.count == 1 and c.cost <= 3)
    dear = next(n for n, c in cards.items()
                if c.type == CardType.TROOP and c.count == 1 and c.cost >= 6)

    assert resolve_spawn(CardType.TROOP, 1, cards, [cheap, dear])[0] == ""
    assert resolve_spawn(CardType.TROOP, 1, cards, [cheap, dear],
                         elixir_max=float(cards[cheap].cost))[0] == cheap


def test_ambiguity_banks_nothing_and_says_why(cards):
    """A wrong label banked to disk corrupts every model trained after it,
    which is worse than any single wrong identity in a match."""
    pair = [n for n, c in cards.items()
            if c.type == CardType.TROOP and c.count == 1 and c.cost == 3][:2]
    if len(pair) < 2:
        pytest.skip("card table has no two same-cost single troops")

    card, reason = resolve_spawn(CardType.TROOP, 1, cards, pair,
                                 elixir_max=10.0)

    assert card == ""
    assert "ambiguous" in reason


def test_no_prior_banks_nothing(cards):
    card, reason = resolve_spawn(CardType.TROOP, 1, cards, [])

    assert card == ""
    assert "no deck prior" in reason


def test_a_kind_absent_from_the_prior_banks_nothing(cards):
    card, reason = resolve_spawn(CardType.BUILDING, 1, cards, ["knight", "archers"])

    assert card == ""
    assert "no building" in reason


# ------------------------------------------------------------- the labeller


def test_a_settled_spawn_is_banked(cards, frame):
    labeler = SelfLabeler(cards, CONFIG)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    spawn = _detection(kind=CardType.BUILDING)

    assert labeler.observe(frame, [spawn], 0.0, tracker) == []   # spawn frame
    assert labeler.observe(frame, [spawn], 0.1, tracker) == []   # still settling
    banked = labeler.observe(frame, [spawn], 0.2, tracker)

    assert [b.card for b in banked] == ["cannon"]
    assert banked[0].annotations[0].card == "cannon"
    assert banked[0].annotations[0].kind == CardType.BUILDING.value


def test_the_deploy_frame_is_not_the_one_banked(cards, frame):
    """The spawn frame is deploy VFX. Banking it teaches the detector that
    every card looks like a puff of smoke for its first half-second."""
    labeler = SelfLabeler(cards, AutoLabelConfig(settle_frames=5))
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    spawn = _detection(kind=CardType.BUILDING)

    banked = [labeler.observe(frame, [spawn], i * 0.05, tracker) for i in range(4)]

    assert all(b == [] for b in banked)


def test_a_moving_unit_is_not_a_new_spawn(cards, frame):
    """Every frame would otherwise bank the same unit again, and the store
    would fill with one Knight."""
    labeler = SelfLabeler(cards, CONFIG)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])

    banked = []
    for i in range(10):
        moved = _detection(x=100.0, y=200.0 - i * 8.0, kind=CardType.BUILDING)
        banked.extend(labeler.observe(frame, [moved], i * 0.05, tracker))

    assert len(banked) == 1, f"banked {len(banked)} times for one deploy"


def test_a_unit_that_dies_before_settling_is_not_banked(cards, frame):
    labeler = SelfLabeler(cards, AutoLabelConfig(settle_frames=4))
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    spawn = _detection(kind=CardType.BUILDING)

    labeler.observe(frame, [spawn], 0.0, tracker)
    for i in range(1, 6):
        assert labeler.observe(frame, [], i * 0.05, tracker) == []


def test_friendly_units_are_not_labelled_this_way(cards, frame):
    """Our own deploys are already known exactly -- the hand cycle names
    them. Running them through a deduction could only be less certain."""
    labeler = SelfLabeler(cards, CONFIG)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    ours = _detection(kind=CardType.BUILDING, team="friendly")

    banked = [labeler.observe(frame, [ours], i * 0.05, tracker) for i in range(5)]

    assert all(b == [] for b in banked)


def test_weak_detections_are_not_banked(cards, frame):
    """The box would be unreliable even where the deduced name is right."""
    labeler = SelfLabeler(cards, AutoLabelConfig(settle_frames=2, min_score=0.8))
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    faint = _detection(kind=CardType.BUILDING, score=0.5)

    banked = [labeler.observe(frame, [faint], i * 0.05, tracker) for i in range(5)]

    assert all(b == [] for b in banked)


def test_a_mixed_kind_clump_is_not_deduced_from(cards, frame):
    """Two things that happened to land near each other are not one deploy,
    and reading a spawn count off them would bank a wrong box."""
    labeler = SelfLabeler(cards, CONFIG)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    mixed = [_detection(x=100.0, kind=CardType.BUILDING),
             _detection(x=130.0, kind=CardType.TROOP)]

    banked = [labeler.observe(frame, mixed, i * 0.05, tracker) for i in range(5)]

    assert all(b == [] for b in banked)


def test_without_a_tracker_nothing_is_deduced(cards, frame):
    """No prior, no deduction. The point is that the constraints do the
    work, not that the detector's own guess gets written down as truth."""
    labeler = SelfLabeler(cards, CONFIG)
    spawn = _detection(kind=CardType.BUILDING, card="cannon")

    banked = [labeler.observe(frame, [spawn], i * 0.05) for i in range(5)]

    assert all(b == [] for b in banked)


def test_reset_drops_pending_work(cards, frame):
    labeler = SelfLabeler(cards, AutoLabelConfig(settle_frames=4))
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    labeler.observe(frame, [_detection(kind=CardType.BUILDING)], 0.0, tracker)

    labeler.reset()

    assert labeler.observe(frame, [], 0.2, tracker) == []


# ------------------------------------------------------------------- store


def test_store_writes_a_manifest_the_trainer_can_read(cards, frame, tmp_path):
    """Same format `synth.build_dataset` emits, so a training run mixes
    synthetic and banked data by pointing at both."""
    store = LabelStore(tmp_path)
    labeler = SelfLabeler(cards, CONFIG, store=store)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    spawn = _detection(kind=CardType.BUILDING)
    for i in range(3):
        labeler.observe(frame, [spawn], i * 0.05, tracker)

    path = store.flush()

    manifest = json.loads(path.read_text())
    assert manifest["cards"] == ["cannon"]
    assert len(manifest["scenes"]) == 1
    assert (tmp_path / manifest["scenes"][0]["image"]).exists()
    assert manifest["scenes"][0]["annotations"][0]["card"] == "cannon"


def test_banked_frames_load_through_the_training_dataset(cards, frame, tmp_path):
    """The loop only closes if the trainer can actually consume what the
    labeller banked."""
    from src.live.detector import DetectorConfig
    from src.live.train_detector import SceneDataset

    store = LabelStore(tmp_path)
    labeler = SelfLabeler(cards, CONFIG, store=store)
    tracker = _Tracker(["cannon", "knight", "archers", "fireball"])
    for i in range(3):
        labeler.observe(frame, [_detection(kind=CardType.BUILDING)], i * 0.05, tracker)
    manifest = store.flush()

    config = DetectorConfig(cards=json.loads(manifest.read_text())["cards"],
                            input_size=(64, 64), width=8)
    dataset = SceneDataset(manifest, config)
    image, targets, annotations, size = dataset[0]

    assert image.shape == (3, 64, 64)
    assert targets["heatmap"].shape[0] == 1
    assert annotations[0]["card"] == "cannon"
