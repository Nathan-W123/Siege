"""The learned detector: targets, loss, decode, and the live adapter.

The load-bearing test here is `test_overfits_a_single_scene`. Shape tests
prove the tensors line up; only closing the loop -- encode targets, train,
decode, and find the thing that was drawn -- proves the encoding and the
decoding agree about what a centre, an offset and a size mean. Those two
halves disagreeing is the classic silent failure of a detector, and it
looks exactly like a model that trains and predicts nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.live.detector import (
    KINDS,
    TEAMS,
    DetectedEntity,
    DetectorConfig,
    build_detector,
    decode,
    detector_loss,
    encode_targets,
    focal_loss,
    gaussian_radius,
    load_checkpoint,
    save_checkpoint,
    to_perceived,
)
from src.simulator.constants import CardType

CARDS = ["knight", "cannon"]


def _config(**overrides):
    base = dict(cards=list(CARDS), input_size=(64, 64), stride=4, width=8)
    base.update(overrides)
    return DetectorConfig(**base)


def _annotation(card="knight", kind="troop", team="hostile", box=(20, 20, 36, 44)):
    x0, y0, x1, y1 = box
    return {"card": card, "kind": kind, "team": team,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _batch(targets):
    return {k: torch.from_numpy(v).unsqueeze(0) for k, v in targets.items()}


# ------------------------------------------------------------------ targets


def test_radius_grows_with_the_box():
    """A fixed radius would punish a Golem's centre as hard as a
    Skeleton's, when the same pixel error means very different things."""
    assert gaussian_radius(40, 40) > gaussian_radius(8, 8)


def test_peak_lands_on_the_box_centre():
    config = _config()
    targets = encode_targets([_annotation(box=(20, 20, 36, 44))], (64, 64), config)

    peak = np.unravel_index(targets["heatmap"][0].argmax(), targets["heatmap"][0].shape)

    assert peak == (8, 7)          # centre (28, 32) / stride 4
    assert targets["heatmap"][0].max() == pytest.approx(1.0)


def test_only_the_annotated_card_channel_is_lit():
    config = _config()
    targets = encode_targets([_annotation(card="cannon")], (64, 64), config)

    assert targets["heatmap"][CARDS.index("cannon")].max() == pytest.approx(1.0)
    assert not targets["heatmap"][CARDS.index("knight")].any()


def test_overlapping_same_card_peaks_do_not_stack():
    """Score has to mean confidence, not "how crowded is it here"."""
    config = _config()
    close = [_annotation(box=(20, 20, 36, 44)), _annotation(box=(21, 21, 37, 45))]

    targets = encode_targets(close, (64, 64), config)

    assert targets["heatmap"].max() == pytest.approx(1.0)


def test_team_and_kind_are_recorded_at_the_centre():
    config = _config()
    targets = encode_targets([_annotation(kind="building", team="friendly")],
                             (64, 64), config)

    iy, ix = np.unravel_index(targets["mask"].argmax(), targets["mask"].shape)
    assert TEAMS[targets["team"][iy, ix]] == "friendly"
    assert KINDS[targets["kind"][iy, ix]] == CardType.BUILDING


def test_boxes_rescale_from_the_scene_resolution():
    """A dataset rendered at one resolution must train a model at another
    without anything downstream knowing the difference."""
    config = _config()

    full = encode_targets([_annotation(box=(20, 20, 36, 44))], (64, 64), config)
    half = encode_targets([_annotation(box=(40, 40, 72, 88))], (128, 128), config)

    assert np.array_equal(full["heatmap"], half["heatmap"])
    assert np.allclose(full["size"], half["size"])


def test_unknown_cards_are_skipped_not_guessed():
    """A dataset holding a card the model has no channel for must not have
    it silently folded into channel zero."""
    config = _config()

    targets = encode_targets([_annotation(card="mega_knight")], (64, 64), config)

    assert not targets["mask"].any()


def test_degenerate_boxes_are_skipped():
    config = _config()
    assert not encode_targets([_annotation(box=(20, 20, 20, 44))],
                              (64, 64), config)["mask"].any()


# --------------------------------------------------------------------- loss


def test_focal_loss_rewards_a_confident_correct_peak():
    target = torch.zeros(1, 2, 8, 8)
    target[0, 0, 4, 4] = 1.0
    good = torch.full((1, 2, 8, 8), -4.0)
    good[0, 0, 4, 4] = 4.0

    assert float(focal_loss(good, target)) < float(focal_loss(torch.zeros(1, 2, 8, 8), target))


def test_an_empty_scene_still_produces_a_loss():
    """An empty arena is a legitimate scene; its whole contribution is
    "everything here is background"."""
    target = torch.zeros(1, 2, 8, 8)

    assert float(focal_loss(torch.zeros(1, 2, 8, 8), target)) > 0.0


def test_loss_reports_its_parts():
    config = _config()
    model = build_detector(config)
    targets = _batch(encode_targets([_annotation()], (64, 64), config))

    total, parts = detector_loss(model(torch.zeros(1, 3, 64, 64)), targets)

    assert set(parts) == {"heatmap", "size", "offset", "team", "kind"}
    assert float(total.detach()) > 0.0


def test_a_scene_with_no_annotations_does_not_crash_the_loss():
    config = _config()
    model = build_detector(config)
    targets = _batch(encode_targets([], (64, 64), config))

    total, _ = detector_loss(model(torch.zeros(1, 3, 64, 64)), targets)

    assert torch.isfinite(total)


# -------------------------------------------------------------------- model


def test_forward_shapes_match_the_configured_stride():
    config = _config()
    out_w, out_h = config.out_size

    outputs = build_detector(config)(torch.zeros(2, 3, 64, 64))

    # Cards plus the trailing "entity, card not known" channel.
    assert outputs["heatmap"].shape == (2, len(CARDS) + 1, out_h, out_w)
    assert outputs["size"].shape == (2, 2, out_h, out_w)
    assert outputs["team"].shape == (2, len(TEAMS), out_h, out_w)
    assert outputs["kind"].shape == (2, len(KINDS), out_h, out_w)


def test_heatmap_head_starts_pessimistic():
    """Focal loss on a head initialised at 0.5 spends its first epochs
    driving an almost-empty heatmap down, and often collapses first."""
    model = build_detector(_config())

    assert torch.sigmoid(model.heatmap.bias).max() < 0.2


# ----------------------------------------------------- the end-to-end check


def test_overfits_a_single_scene():
    """Encode, train, decode, and find what was drawn.

    If the target encoder and the decoder disagree about what a centre, an
    offset or a size means, this is the only test that notices -- the model
    still trains and the loss still falls.
    """
    config = _config(score_threshold=0.2, identity_threshold=0.2)
    model = build_detector(config)
    annotations = [_annotation(card="cannon", kind="building", team="friendly",
                               box=(20, 24, 36, 48))]
    targets = _batch(encode_targets(annotations, (64, 64), config))

    # A distinctive image, so the model has something to key on.
    image = torch.zeros(1, 3, 64, 64)
    image[0, :, 24:48, 20:36] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        optimizer.zero_grad()
        loss, _ = detector_loss(model(image), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        found = decode(model(image), config)[0]

    assert found, "trained to convergence on one scene and detected nothing"
    best = found[0]
    assert best.card == "cannon"
    assert best.kind == CardType.BUILDING
    assert best.team == "friendly"
    assert best.x0 == pytest.approx(20, abs=4)
    assert best.y1 == pytest.approx(48, abs=4)


# ------------------------------------------------------------------ decode


def _outputs_with_peak(config, score=0.9, cls=0, team=1, kind=0):
    out_w, out_h = config.out_size
    logit = float(np.log(score / (1 - score)))
    heatmap = torch.full((1, len(CARDS), out_h, out_w), -10.0)
    heatmap[0, cls, 5, 6] = logit
    size = torch.zeros(1, 2, out_h, out_w)
    size[0, :, 5, 6] = torch.tensor([16.0, 24.0])
    offset = torch.zeros(1, 2, out_h, out_w)
    team_t = torch.zeros(1, len(TEAMS), out_h, out_w)
    team_t[0, team, 5, 6] = 5.0
    kind_t = torch.zeros(1, len(KINDS), out_h, out_w)
    kind_t[0, kind, 5, 6] = 5.0
    return {"heatmap": heatmap, "size": size, "offset": offset,
            "team": team_t, "kind": kind_t}


def test_decode_recovers_the_box():
    config = _config()

    found = decode(_outputs_with_peak(config), config)[0]

    assert len(found) == 1
    assert found[0].x0 == pytest.approx(6 * 4 - 8)
    assert found[0].y1 == pytest.approx(5 * 4 + 12)


def test_weak_peaks_are_not_reported():
    config = _config(score_threshold=0.5)

    assert decode(_outputs_with_peak(config, score=0.3), config)[0] == []


def test_identity_abstains_but_kind_and_team_survive():
    """The reason these are separate heads. `ShadowEngine` needs the kind
    far more than it needs the card name -- kind decides which observation
    channel the entity lands in at all."""
    config = _config(score_threshold=0.2, identity_threshold=0.8)

    found = decode(_outputs_with_peak(config, score=0.4, kind=1, team=1), config)[0]

    assert len(found) == 1
    assert found[0].card == ""
    assert found[0].kind == CardType.BUILDING
    assert found[0].team == "hostile"


# ------------------------------------------------------------ live adapter


def test_perceived_units_project_from_the_feet(arena):
    """The box centre would put a tall unit a tile or more behind where it
    stands, and placement decisions are made in tiles."""
    from src.live.homography import Homography
    from tests.live_frames import render_empty

    _, meta = render_empty(arena)
    homography = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})

    tall = DetectedEntity(card="giant", kind=CardType.TROOP, team="hostile",
                          score=0.9, identity_score=0.9,
                          x0=260.0, y0=500.0, x1=300.0, y1=620.0)

    units = to_perceived([tall], homography, arena)

    assert len(units) == 1
    feet_tile = homography.pixel_to_tile(280.0, 620.0)
    assert units[0].tile_y == pytest.approx(feet_tile[1])


def test_spell_detections_do_not_become_units(arena):
    """Spells come through `src.live.spells`. One arriving here would
    materialise a troop that never leaves the arena."""
    from src.live.homography import Homography
    from tests.live_frames import render_empty

    _, meta = render_empty(arena)
    homography = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})

    spell = DetectedEntity(card="fireball", kind=CardType.SPELL, team="hostile",
                           score=0.9, identity_score=0.9,
                           x0=260.0, y0=500.0, x1=300.0, y1=540.0)

    assert to_perceived([spell], homography, arena) == []


def test_hp_is_reported_as_unknown_not_guessed(arena):
    """The detector is not trained to read a bar's fill. A fabricated
    fraction would flow into the trade arithmetic the policy defends with."""
    from src.live.homography import Homography
    from tests.live_frames import render_empty

    _, meta = render_empty(arena)
    homography = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})

    unit = DetectedEntity(card="knight", kind=CardType.TROOP, team="hostile",
                          score=0.9, identity_score=0.9,
                          x0=260.0, y0=560.0, x1=300.0, y1=620.0)

    assert to_perceived([unit], homography, arena)[0].hp_confident is False


# -------------------------------------------------------------- checkpoints


def test_checkpoint_carries_its_card_list(tmp_path):
    """The card list defines the heatmap's channel count and its order.
    Loading weights against a differently ordered list silently permutes
    every class, and nothing downstream would notice."""
    config = _config()
    model = build_detector(config)

    save_checkpoint(tmp_path / "d.pt", model, config)
    loaded, loaded_config = load_checkpoint(tmp_path / "d.pt")

    assert loaded_config.cards == CARDS
    assert loaded_config.input_size == (64, 64)
    with torch.no_grad():
        image = torch.rand(1, 3, 64, 64)
        assert torch.allclose(loaded(image)["heatmap"], model.eval()(image)["heatmap"])


# ----------------------------------------------------- the unnamed channel


def _unsupervised(**overrides):
    a = _annotation(**overrides)
    a["identity_supervised"] = False
    return a


def test_an_unsupervised_example_trains_the_unnamed_channel(): 
    """A friendly sprite retinted into an enemy has the right silhouette and
    the wrong face. It may teach "an entity is here"; it may not teach which
    card, because a real enemy of that card shows a front."""
    config = _config()

    targets = encode_targets([_unsupervised(card="knight")], (64, 64), config)

    assert targets["heatmap"][config.unnamed_channel].max() == pytest.approx(1.0)
    assert not targets["heatmap"][CARDS.index("knight")].any()


def test_a_supervised_example_still_trains_its_card_channel():
    config = _config()

    targets = encode_targets([_annotation(card="knight")], (64, 64), config)

    assert targets["heatmap"][CARDS.index("knight")].max() == pytest.approx(1.0)
    assert not targets["heatmap"][config.unnamed_channel].any()
    assert not targets["identity_ignore"].any()


def test_box_team_and_kind_are_supervised_either_way():
    """Everything that transfers across facing still gets taught. Only the
    card name is withheld."""
    config = _config()

    supervised = encode_targets([_annotation()], (64, 64), config)
    withheld = encode_targets([_unsupervised()], (64, 64), config)

    assert np.array_equal(supervised["mask"], withheld["mask"])
    assert np.allclose(supervised["size"], withheld["size"])
    assert np.array_equal(supervised["team"], withheld["team"])
    assert np.array_equal(supervised["kind"], withheld["kind"])


def test_the_card_channels_are_not_scored_where_identity_is_withheld():
    """The subtle half of the fix.

    Leaving those cells in the loss as ordinary background would teach "an
    enemy facing away is not a Knight" — a claim about facing, dressed up as
    one about identity. So the loss must not move when the card channels'
    predictions there change.
    """
    config = _config()
    targets = _batch(encode_targets([_unsupervised(card="knight")], (64, 64), config))
    out_w, out_h = config.out_size

    # Only inside the withheld region. Outside it the card channels are
    # ordinary background and must still be scored as such.
    withheld = targets["identity_ignore"][0].bool()
    assert withheld.any()

    def outputs(card_logit):
        heat = torch.full((1, config.n_heatmap, out_h, out_w), -4.0)
        heat[0, CARDS.index("knight")][withheld] = card_logit
        return {"heatmap": heat,
                "size": torch.zeros(1, 2, out_h, out_w),
                "offset": torch.zeros(1, 2, out_h, out_w),
                "team": torch.zeros(1, len(TEAMS), out_h, out_w),
                "kind": torch.zeros(1, len(KINDS), out_h, out_w)}

    quiet, _ = detector_loss(outputs(-4.0), targets)
    shouting, _ = detector_loss(outputs(6.0), targets)

    assert float(quiet.detach()) == pytest.approx(float(shouting.detach()), rel=1e-5)


def test_an_unnamed_peak_decodes_to_a_real_box_with_no_card():
    """What the bootstrap detector gives `autolabel` to work with: a found
    enemy with a deliberately empty name. The tracker supplies the name."""
    config = _config(score_threshold=0.2)
    out_w, out_h = config.out_size
    heat = torch.full((1, config.n_heatmap, out_h, out_w), -10.0)
    heat[0, config.unnamed_channel, 5, 6] = 4.0
    size = torch.zeros(1, 2, out_h, out_w)
    size[0, :, 5, 6] = torch.tensor([16.0, 24.0])
    team = torch.zeros(1, len(TEAMS), out_h, out_w)
    team[0, TEAMS.index("hostile"), 5, 6] = 5.0
    kind = torch.zeros(1, len(KINDS), out_h, out_w)
    kind[0, KINDS.index(CardType.BUILDING), 5, 6] = 5.0

    found = decode({"heatmap": heat, "size": size,
                    "offset": torch.zeros(1, 2, out_h, out_w),
                    "team": team, "kind": kind}, config)[0]

    assert len(found) == 1
    assert found[0].card == ""
    assert found[0].team == "hostile"
    assert found[0].kind == CardType.BUILDING
    assert found[0].x1 - found[0].x0 == pytest.approx(16.0)
