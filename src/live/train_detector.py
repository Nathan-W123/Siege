"""Train the entity detector on composited scenes.

Run it after a harvest and a dataset build::

    python -m src.live.train_detector --manifest data/synth/manifest.json \\
        --out checkpoints/detector.pt --epochs 20

The metric worth watching is not the loss
-----------------------------------------
Detection losses fall smoothly while the model is still finding nothing:
the heatmap term is dominated by the enormous number of correctly-predicted
background cells, and it keeps improving long after the few cells that
matter have stopped. So evaluation here reports **centre recall** and
**identity accuracy among recalled units** alongside the loss, and those
are the numbers that say whether the thing works.

They are also the two numbers that map onto how the detector fails in play.
Missed recall is a unit the policy never sees, which is a blind spot.
Wrong identity on a recalled unit is a unit the policy sees with the wrong
stats, which is a bad trade. The second is recoverable -- `ShadowEngine`
keeps the kind and the position even when the card name is blank -- and the
first is not, which is why the detector abstains on identity rather than
on the detection.

Feeding the numbers back
------------------------
`src/agent/obs_noise.py` asks, in its own docstring, to be calibrated from
measured detections rather than guessed at: "Once #34 produces real
detections, measure `p_miss` and positional error on recorded frames and fit
these numbers to them." The recall reported here is `1 - p_miss` on
synthetic scenes, which is the first honest estimate available, and it
should be re-measured on real captures before a policy is trained against it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.live.detector import (
    DetectorConfig,
    build_detector,
    decode,
    detector_loss,
    encode_targets,
    save_checkpoint,
)


class SceneDataset:
    """Composited scenes and their targets, read from a `synth` manifest.

    Targets are encoded on demand rather than precomputed: at stride 4 the
    heatmap alone is one float per card per cell, so a hundred-card roster
    over a large dataset is tens of gigabytes of tensors to cache in order
    to save a few milliseconds of numpy per sample.
    """

    def __init__(self, manifest_path: Path | str, config: DetectorConfig):
        manifest_path = Path(manifest_path)
        self.root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        self.scenes = manifest["scenes"]
        self.config = config

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int):
        import torch

        entry = self.scenes[index]
        image = Image.open(self.root / entry["image"]).convert("RGB")
        scene_size = image.size
        resized = image.resize(self.config.input_size, Image.BILINEAR)
        pixels = np.asarray(resized, np.float32).transpose(2, 0, 1) / 255.0
        targets = encode_targets(entry["annotations"], scene_size, self.config)
        return (torch.from_numpy(pixels),
                {k: torch.from_numpy(v) for k, v in targets.items()},
                entry["annotations"], scene_size)


def _collate(batch):
    import torch

    images = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return images, targets, [b[2] for b in batch], [b[3] for b in batch]


def evaluate(model, dataset, config: DetectorConfig, indices, tolerance: float = 8.0):
    """Centre recall and identity accuracy on a held-out split.

    A ground-truth box counts as recalled when some detection's centre lands
    within `tolerance` input-space pixels of its own. Centre distance rather
    than IoU on purpose: at stride 4 the box regression is the coarsest part
    of the output, and IoU would fold a size error the policy does not care
    about into a recall number that it does. Position is what becomes a tile.
    """
    import torch

    model.eval()
    recalled = matched = correct = total = 0
    with torch.no_grad():
        for index in indices:
            image, _, annotations, scene_size = dataset[index]
            found = decode(model(image.unsqueeze(0)), config)[0]
            sx = config.input_size[0] / float(scene_size[0])
            sy = config.input_size[1] / float(scene_size[1])
            for a in annotations:
                if a["card"] not in config.cards:
                    continue
                total += 1
                cx = (a["x0"] + a["x1"]) / 2.0 * sx
                cy = (a["y0"] + a["y1"]) / 2.0 * sy
                near = [d for d in found
                        if np.hypot((d.x0 + d.x1) / 2.0 - cx,
                                    (d.y0 + d.y1) / 2.0 - cy) <= tolerance]
                if not near:
                    continue
                recalled += 1
                best = max(near, key=lambda d: d.score)
                if best.card:
                    matched += 1
                    correct += int(best.card == a["card"])
    return {
        "recall": recalled / total if total else 0.0,
        "identity_accuracy": correct / matched if matched else 0.0,
        "identity_reported": matched / recalled if recalled else 0.0,
        "boxes": total,
    }


def train(
    manifest: Path | str,
    out: Path | str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    input_size: tuple[int, int] = (256, 448),
    width: int = 32,
    holdout: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    log: Path | str | None = None,
):
    import torch
    from torch.utils.data import DataLoader, Subset

    manifest_path = Path(manifest)
    cards = json.loads(manifest_path.read_text())["cards"]
    config = DetectorConfig(cards=list(cards), input_size=tuple(input_size), width=width)
    dataset = SceneDataset(manifest_path, config)

    # Split by index rather than by shuffling the manifest, so the same seed
    # gives the same holdout across runs and two runs stay comparable.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    n_hold = max(1, int(len(dataset) * holdout))
    hold_idx, train_idx = order[:n_hold].tolist(), order[n_hold:].tolist()

    model = build_detector(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                        shuffle=True, collate_fn=_collate)

    rows = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, targets, _, _ in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            optimizer.zero_grad()
            loss, _ = detector_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        schedule.step()

        metrics = evaluate(model, dataset, config, hold_idx)
        row = {"epoch": epoch, "loss": running / max(1, len(loader)), **metrics}
        rows.append(row)
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  "
              f"recall {row['recall']:.3f}  identity {row['identity_accuracy']:.3f}")

    save_checkpoint(out, model, config)
    if log is not None:
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return model, config, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 448),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    train(args.manifest, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, input_size=tuple(args.input_size), width=args.width,
          holdout=args.holdout, seed=args.seed, device=args.device, log=args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
