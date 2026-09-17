"""A learned entity detector: one model in place of thresholds and templates.

What it replaces
----------------
Two things, and for two different reasons.

`vision.py` finds units by masking team-tinted health bars. Its own
docstring admits the constants are "Starting points, not measurements", and
they are: a hue window is an absolute statement about colour, so every arena
skin, day/night variant and display colour profile moves it. Worse, a bar is
a *proxy*. Swarms merge their bars into one blob, a full-HP bar is sometimes
not drawn at all, and the fixed bar-to-feet offset is one constant for units
that range from a Skeleton to a Golem.

`identify.py` names the sprite under the bar by template correlation. It is
a good design given no training data, and it is bounded by exactly that: a
template library is specific to one skin at one resolution, and building one
per card is the hand-labelling cost this whole line of work exists to remove.

A detector trained on `synth.py` scenes has neither problem. It never sees an
absolute colour it must match, because appearance randomization made colour
uninformative during training. It finds units rather than their bars, so
swarms and bar-less units are ordinary cases. And its labels cost nothing,
so retraining for a new skin or a new season is a compute cost, not a
person's afternoon.

Why anchor-free, and why this small
-----------------------------------
The live budget is about half a second per decision, shared with a policy
forward pass, on whatever machine the game is running on. That rules out
anything with a heavy backbone, and it makes anchor boxes and NMS -- both
of which cost more the more units are on screen, which is exactly when the
budget is tightest -- a bad trade. A CenterNet-style heatmap gives a fixed
cost: peaks are found by one max-pool, and the number of units on screen
does not change the arithmetic.

Anchor-free also fits the subject. Arena units occupy a narrow band of
apparent sizes, set by the perspective, and a centre-plus-size
parameterization spends no capacity on the box shapes that never occur.

Three heads, because abstention should be partial
-------------------------------------------------
Card identity, team and kind are predicted separately rather than as one
joint class. They are different problems with very different difficulty:
which of a hundred cards a sprite is, is genuinely hard; whether it is
friendly, and whether it is a building, are easy and stay accurate long
after identity has given up. Folding them into one label would throw away
the easy answers whenever the hard one is uncertain -- and the easy answers
are the ones `ShadowEngine` needs most, since kind decides which observation
channel the entity lands in at all.

Nothing here reads game memory or network traffic; it only looks at pixels
that are already on screen. See CLAUDE.md, "On-Screen Visual Perception".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.simulator.constants import CardType

KINDS = (CardType.TROOP, CardType.BUILDING, CardType.SPELL)
KIND_TO_ID = {k.value: i for i, k in enumerate(KINDS)}
TEAMS = ("friendly", "hostile")


@dataclass
class DetectorConfig:
    """Geometry and capacity. Stored in the checkpoint, never re-derived."""

    cards: list[str] = field(default_factory=list)
    # Input is letterbox-free: scenes are resized outright, because the
    # capture aspect ratio is fixed by the game client and a letterbox would
    # spend pixels on bars that are never there.
    input_size: tuple[int, int] = (256, 448)     # (width, height)
    stride: int = 4
    width: int = 32                              # base channel count
    # Below this peak score a detection is not reported at all.
    score_threshold: float = 0.30
    # ...and below this, the box is reported with its card blanked. Kind and
    # team survive, which is the whole reason they are separate heads.
    identity_threshold: float = 0.55

    @property
    def n_cards(self) -> int:
        return len(self.cards)

    @property
    def out_size(self) -> tuple[int, int]:
        return self.input_size[0] // self.stride, self.input_size[1] // self.stride


@dataclass(frozen=True)
class DetectedEntity:
    """One detection, in input-image pixels.

    `card` is "" when the identity head was not confident enough. That is a
    real answer and the reason `kind` is carried beside it rather than being
    read off the card name.
    """

    card: str
    kind: CardType
    team: str
    score: float
    identity_score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom-centre: where the unit meets the ground, which is the
        point the homography turns into an arena tile. The box centre would
        project a tall unit a tile or more behind where it stands."""
        return ((self.x0 + self.x1) / 2.0, self.y1)


# ------------------------------------------------------------------ targets


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    """CenterNet's radius: how far a predicted centre may drift and still
    overlap the true box by `min_overlap`. Splatting a fixed radius instead
    would punish a Golem's centre as hard as a Skeleton's, when the same
    pixel error means very different things for the two."""
    a = 1.0
    b = height + width
    c = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    discriminant = max(0.0, b * b - 4 * a * c)
    return max(1, int((b - np.sqrt(discriminant)) / (2 * a)))


def _splat(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Draw an unnormalized Gaussian, keeping the running maximum.

    Maximum rather than sum: two units of the same card standing close
    together must not build a peak taller than either one, or the score
    stops meaning "confidence" and starts meaning "how crowded is it here".
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(xs * xs + ys * ys) / (2.0 * sigma * sigma))
    h, w = heatmap.shape
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    patch = kernel[y0 - (cy - radius):y1 - (cy - radius),
                   x0 - (cx - radius):x1 - (cx - radius)]
    np.maximum(heatmap[y0:y1, x0:x1], patch, out=heatmap[y0:y1, x0:x1])


def encode_targets(annotations, scene_size, config: DetectorConfig):
    """Annotations -> the five target tensors, as numpy.

    `scene_size` is the size the annotations' boxes are in; boxes are scaled
    to the network's input size here rather than by the caller, so a dataset
    rendered at one resolution trains a model at another without anything
    downstream having to know.
    """
    card_to_id = {name: i for i, name in enumerate(config.cards)}
    out_w, out_h = config.out_size
    sx = config.input_size[0] / float(scene_size[0])
    sy = config.input_size[1] / float(scene_size[1])

    heatmap = np.zeros((max(1, config.n_cards), out_h, out_w), np.float32)
    size = np.zeros((2, out_h, out_w), np.float32)
    offset = np.zeros((2, out_h, out_w), np.float32)
    team = np.zeros((out_h, out_w), np.int64)
    kind = np.zeros((out_h, out_w), np.int64)
    mask = np.zeros((out_h, out_w), np.float32)

    for a in annotations:
        card = a["card"] if isinstance(a, dict) else a.card
        if card not in card_to_id:
            continue
        x0, y0 = (a["x0"] if isinstance(a, dict) else a.x0) * sx, \
                 (a["y0"] if isinstance(a, dict) else a.y0) * sy
        x1, y1 = (a["x1"] if isinstance(a, dict) else a.x1) * sx, \
                 (a["y1"] if isinstance(a, dict) else a.y1) * sy
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1.0 or bh <= 1.0:
            continue
        cx, cy = (x0 + x1) / 2.0 / config.stride, (y0 + y1) / 2.0 / config.stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < out_w and 0 <= iy < out_h):
            continue

        _splat(heatmap[card_to_id[card]], ix, iy,
               gaussian_radius(bh / config.stride, bw / config.stride))
        size[:, iy, ix] = (bw, bh)
        offset[:, iy, ix] = (cx - ix, cy - iy)
        team[iy, ix] = TEAMS.index(a["team"] if isinstance(a, dict) else a.team)
        kind[iy, ix] = KIND_TO_ID[a["kind"] if isinstance(a, dict) else a.kind]
        mask[iy, ix] = 1.0

    return {"heatmap": heatmap, "size": size, "offset": offset,
            "team": team, "kind": kind, "mask": mask}


# -------------------------------------------------------------------- model


def _build_model(config: DetectorConfig):
    """Defined inside a function so importing this module does not import
    torch. `src.live` is on the latency-sensitive path and the rest of it
    is numpy; see the ONNX startup note in the sibling chess project for
    what a stray top-level torch import costs."""
    import torch
    from torch import nn

    w = config.width

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class EntityDetector(nn.Module):
        """Small U-net trunk, four heads at stride 4.

        The trunk goes down to stride 16 and back up rather than staying at
        stride 4 throughout, because identity needs context a 4-pixel
        receptive field does not have -- two cards can share a silhouette
        and differ only in what is drawn around them.
        """

        def __init__(self):
            super().__init__()
            self.enc1 = block(3, w, stride=2)          # /2
            self.enc2 = block(w, w * 2, stride=2)      # /4
            self.enc3 = block(w * 2, w * 4, stride=2)  # /8
            self.enc4 = block(w * 4, w * 4, stride=2)  # /16
            self.up3 = block(w * 8, w * 4)
            self.up2 = block(w * 6, w * 2)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

            def head(cout, bias=None):
                layer = nn.Conv2d(w * 2, cout, 1)
                if bias is not None:
                    nn.init.constant_(layer.bias, bias)
                return layer

            # -2.19 puts the initial sigmoid near 0.1. Focal loss on a head
            # initialised at 0.5 spends its first epochs driving almost
            # every cell of an almost-empty heatmap down, and frequently
            # collapses to predicting nothing at all before it recovers.
            self.heatmap = head(max(1, config.n_cards), bias=-2.19)
            self.size = head(2)
            self.offset = head(2)
            self.team = head(len(TEAMS))
            self.kind = head(len(KINDS))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            d3 = self.up3(torch.cat([self.upsample(e4), e3], dim=1))
            d2 = self.up2(torch.cat([self.upsample(d3), e2], dim=1))
            return {
                "heatmap": self.heatmap(d2),
                "size": self.size(d2),
                "offset": self.offset(d2),
                "team": self.team(d2),
                "kind": self.kind(d2),
            }

    return EntityDetector()


def build_detector(config: DetectorConfig):
    return _build_model(config)


# --------------------------------------------------------------------- loss


def focal_loss(pred_logits, target):
    """CenterNet's penalty-reduced focal loss.

    Cells near a true centre are not simply negatives — the Gaussian says
    how nearly right they are — so their penalty is scaled by `(1 - t)^4`.
    Treating them as hard negatives fights the splat that was drawn on
    purpose and blurs every centre it is trying to sharpen.
    """
    import torch

    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1.0 - 1e-4)
    positive = target.ge(1.0 - 1e-6).float()
    negative = 1.0 - positive

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, 2.0) * positive
    neg_loss = (-torch.log(1.0 - pred) * torch.pow(pred, 2.0)
                * torch.pow(1.0 - target, 4.0) * negative)

    n = positive.sum()
    if n == 0:
        # A scene with no annotations is legitimate — an empty arena. Its
        # whole contribution is "everything here is background".
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n


def detector_loss(outputs, targets, weights=(1.0, 0.1, 1.0, 0.5, 0.5)):
    """Total loss and its parts.

    Size is weighted down by an order of magnitude because it is regressed
    in pixels while everything else is a probability: left at parity its
    gradients dominate and the heatmap never sharpens.
    """
    import torch.nn.functional as F

    w_hm, w_size, w_off, w_team, w_kind = weights
    mask = targets["mask"]
    n = mask.sum().clamp(min=1.0)
    pick = mask.bool()

    parts = {"heatmap": focal_loss(outputs["heatmap"], targets["heatmap"])}

    # Regressions and classifications are supervised only at true centres.
    # Everywhere else there is no box to predict the size of, and averaging
    # a zero target over the whole map would drag every prediction to zero.
    def at_centres(tensor):
        return tensor.permute(0, 2, 3, 1)[pick]

    if pick.any():
        parts["size"] = F.l1_loss(at_centres(outputs["size"]),
                                  at_centres(targets["size"]), reduction="sum") / n
        parts["offset"] = F.l1_loss(at_centres(outputs["offset"]),
                                    at_centres(targets["offset"]), reduction="sum") / n
        parts["team"] = F.cross_entropy(at_centres(outputs["team"]),
                                        targets["team"][pick], reduction="mean")
        parts["kind"] = F.cross_entropy(at_centres(outputs["kind"]),
                                        targets["kind"][pick], reduction="mean")
    else:
        zero = outputs["size"].sum() * 0.0
        parts.update(size=zero, offset=zero, team=zero, kind=zero)

    total = (w_hm * parts["heatmap"] + w_size * parts["size"]
             + w_off * parts["offset"] + w_team * parts["team"]
             + w_kind * parts["kind"])
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ------------------------------------------------------------------ decode


def decode(outputs, config: DetectorConfig, max_detections: int = 40):
    """Network outputs -> detections for one batch element.

    Peaks are isolated by a 3x3 max-pool rather than by NMS over boxes. It
    costs the same whether the arena holds two units or thirty, which is
    the property that keeps a crowded frame inside the decision budget --
    box NMS gets slower exactly when the board is busiest.
    """
    import torch
    import torch.nn.functional as F

    heatmap = torch.sigmoid(outputs["heatmap"])
    pooled = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    peaks = heatmap * (pooled == heatmap).float()

    batch, n_classes, out_h, out_w = peaks.shape
    results: list[list[DetectedEntity]] = []
    for b in range(batch):
        flat = peaks[b].reshape(-1)
        k = min(max_detections, flat.numel())
        scores, indices = torch.topk(flat, k)
        detections: list[DetectedEntity] = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < config.score_threshold:
                break                          # topk is sorted; the rest are lower
            cls, rest = divmod(index, out_h * out_w)
            iy, ix = divmod(rest, out_w)
            ox, oy = outputs["offset"][b, :, iy, ix].tolist()
            bw, bh = outputs["size"][b, :, iy, ix].tolist()
            cx = (ix + ox) * config.stride
            cy = (iy + oy) * config.stride
            team = TEAMS[int(outputs["team"][b, :, iy, ix].argmax())]
            kind = KINDS[int(outputs["kind"][b, :, iy, ix].argmax())]
            card = config.cards[cls] if cls < config.n_cards else ""
            detections.append(DetectedEntity(
                card=card if score >= config.identity_threshold else "",
                kind=kind, team=team, score=float(score), identity_score=float(score),
                x0=cx - bw / 2.0, y0=cy - bh / 2.0,
                x1=cx + bw / 2.0, y1=cy + bh / 2.0))
        results.append(detections)
    return results


# ------------------------------------------------------------- live adapter


def to_perceived(
    detections,
    homography,
    arena=None,
    scene_size=None,
    config: DetectorConfig | None = None,
):
    """Detections -> `bridge.PerceivedUnit`, projected into arena tiles.

    HP is reported as unknown (full, not confident) rather than guessed.
    The detector is not trained to read a health bar's fill, and a
    fabricated fraction here would flow straight into the shadow engine's
    unit HP and out into the trade arithmetic the policy uses to decide
    whether a defence is worth making. `vision.py`'s bar segmentation stays
    the right tool for that number and composes with this one.
    """
    from src.live.bridge import PerceivedUnit

    units: list[PerceivedUnit] = []
    for d in detections:
        if d.kind == CardType.SPELL:
            continue        # spells go through `src.live.spells`, not here
        px, py = d.feet
        if scene_size is not None and config is not None:
            px *= scene_size[0] / float(config.input_size[0])
            py *= scene_size[1] / float(config.input_size[1])
        try:
            tile_x, tile_y = homography.pixel_to_tile(px, py)
        except ValueError:
            continue
        if arena is not None and not (0.0 <= tile_x < arena.width
                                      and 0.0 <= tile_y < arena.height):
            continue
        units.append(PerceivedUnit(
            card=d.card, tile_x=tile_x, tile_y=tile_y,
            hostile=d.team == "hostile",
            hp_fraction=1.0, hp_confident=False, kind=d.kind))
    return units


# -------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path | str, model, config: DetectorConfig) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": json.dumps({**config.__dict__,
                                      "input_size": list(config.input_size)})}, path)


def load_checkpoint(path: Path | str):
    """Returns (model, config). The config travels with the weights because
    the card list defines the heatmap's channel count — loading weights
    against a differently ordered card list silently permutes every class."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = json.loads(payload["config"])
    raw["input_size"] = tuple(raw["input_size"])
    config = DetectorConfig(**raw)
    model = build_detector(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config
