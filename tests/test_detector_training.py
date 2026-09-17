"""The whole label-free pipeline, end to end, in miniature.

Harvest sprites from rendered deploys, composite them into labelled scenes,
train on those scenes, and evaluate. Nothing in this file writes a label by
hand -- which is the claim the pipeline is making, so it is the claim the
test has to make too.

Everything is deliberately tiny. This is an integration test: it asks
whether the five modules agree about tensor shapes, manifest keys, card
ordering and box conventions. Whether the detector is *accurate* is a
question for real captures and a real training run, and no unit test can
stand in for that.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.live.detector import DetectorConfig, load_checkpoint
from src.live.harvest import HarvestConfig, SpriteLibrary, build_plate, harvest
from src.live.homography import Homography
from src.live.synth import SynthConfig, build_dataset
from src.live.train_detector import SceneDataset, _collate, evaluate, train
from src.live.vision import TEAM_FRIENDLY
from src.simulator.constants import CardType
from tests.live_frames import render_empty, render_unit_with_body

SIZE = (160, 288)
HARVEST = HarvestConfig(min_area=40)


@pytest.fixture()
def dataset_dir(arena, tmp_path):
    """A real dataset, built the way the pipeline builds one."""
    empty, meta = render_empty(arena, size=SIZE)
    plate = build_plate([empty])
    homography = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})

    library = SpriteLibrary()
    for card, kind, tile in (("knight", CardType.TROOP, (5.0, 20.0)),
                             ("cannon", CardType.BUILDING, (13.0, 10.0))):
        frame, _ = render_unit_with_body(arena, tile, team=TEAM_FRIENDLY, size=SIZE)
        library.extend(harvest(frame, plate, card, kind=kind, team=TEAM_FRIENDLY,
                               tile=tile, config=HARVEST))
    assert len(library) >= 2, "fixture harvested nothing to train on"

    build_dataset(tmp_path, empty, library, homography, arena, scenes=12, seed=0,
                  config=SynthConfig(min_units=2, max_units=4))
    return tmp_path


def _config(dataset_dir, **overrides):
    cards = json.loads((dataset_dir / "manifest.json").read_text())["cards"]
    base = dict(cards=list(cards), input_size=(64, 64), stride=4, width=8)
    base.update(overrides)
    return DetectorConfig(**base)


# ------------------------------------------------------------------ dataset


def test_dataset_yields_tensors_the_model_accepts(dataset_dir):
    config = _config(dataset_dir)
    dataset = SceneDataset(dataset_dir / "manifest.json", config)
    out_w, out_h = config.out_size

    image, targets, annotations, scene_size = dataset[0]

    assert image.shape == (3, 64, 64)
    assert targets["heatmap"].shape == (config.n_cards, out_h, out_w)
    assert targets["mask"].shape == (out_h, out_w)
    assert scene_size == SIZE
    assert annotations


def test_targets_survive_the_scene_to_input_rescale(dataset_dir):
    """Scenes are rendered at one size and trained at another. A rescale
    that dropped boxes would train on blank targets and look like a model
    that simply will not converge."""
    config = _config(dataset_dir)
    dataset = SceneDataset(dataset_dir / "manifest.json", config)

    lit = sum(int(dataset[i][1]["mask"].sum()) for i in range(len(dataset)))

    assert lit > 0, "every box fell outside the rescaled heatmap"


def test_collate_stacks_a_batch(dataset_dir):
    config = _config(dataset_dir)
    dataset = SceneDataset(dataset_dir / "manifest.json", config)

    images, targets, annotations, sizes = _collate([dataset[0], dataset[1]])

    assert images.shape[0] == 2
    assert targets["heatmap"].shape[0] == 2
    assert len(annotations) == 2 and len(sizes) == 2


# ----------------------------------------------------------------- training


def test_training_runs_and_the_loss_falls(dataset_dir, tmp_path):
    """The integration check: five modules agreeing well enough to learn."""
    out = tmp_path / "detector.pt"

    _, config, rows = train(dataset_dir / "manifest.json", out, epochs=3,
                            batch_size=4, input_size=(64, 64), width=8,
                            holdout=0.25, seed=0)

    assert len(rows) == 3
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert out.exists()


def test_the_trained_checkpoint_reloads_with_its_cards(dataset_dir, tmp_path):
    out = tmp_path / "detector.pt"
    _, config, _ = train(dataset_dir / "manifest.json", out, epochs=1,
                         batch_size=4, input_size=(64, 64), width=8,
                         holdout=0.25, seed=0)

    model, loaded = load_checkpoint(out)

    assert loaded.cards == config.cards
    assert loaded.input_size == (64, 64)
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 64, 64))["heatmap"].shape[1] == len(loaded.cards)


def test_a_training_log_is_written_when_asked(dataset_dir, tmp_path):
    log = tmp_path / "logs" / "train.csv"

    train(dataset_dir / "manifest.json", tmp_path / "d.pt", epochs=2, batch_size=4,
          input_size=(64, 64), width=8, holdout=0.25, seed=0, log=log)

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 3          # header plus two epochs
    assert "recall" in lines[0]


# --------------------------------------------------------------- evaluation


def test_evaluate_reports_recall_not_just_loss(dataset_dir, tmp_path):
    """Detection loss falls smoothly while the model still finds nothing --
    the background cells dominate it. Recall is the number that says
    whether anything works."""
    out = tmp_path / "detector.pt"
    model, config, _ = train(dataset_dir / "manifest.json", out, epochs=1,
                             batch_size=4, input_size=(64, 64), width=8,
                             holdout=0.25, seed=0)
    dataset = SceneDataset(dataset_dir / "manifest.json", config)

    metrics = evaluate(model, dataset, config, indices=range(len(dataset)))

    assert set(metrics) == {"recall", "identity_accuracy",
                            "identity_reported", "boxes"}
    assert metrics["boxes"] > 0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["identity_accuracy"] <= 1.0
