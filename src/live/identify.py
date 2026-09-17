"""Card identity from sprite patches, constrained by the deck prior (#35).

#34 answers "hostile unit here". This answers "which card". The naive
framing — classify against the full ~108-card roster — is much harder than
the problem actually is, because **a deck is only 8 cards and it reveals
itself over a match**. Once four cards have been played, the candidate set is
a quarter of what a general classifier would face; once all eight are known,
classification is a choice among eight, and anything outside that set is
provably a misread.

That revealed-card state is exactly what the deterministic tracker (#38)
already maintains, so this module *reads* it rather than keeping a second,
divergent copy — `OpponentTracker.candidate_cards()` is the single source.

**Status: the no-training fallback.** This was the first identity path, and
it is kept for the case where no detector checkpoint has been trained yet —
it is cheap, needs no data, and the deck prior does most of the work.
`src/live/detector.py` is the main path now.

The escalation this docstring originally described ("only fine-tune a
detector if measurement shows template matching is the binding constraint")
named the right blocker and drew the wrong conclusion from it. The blocker
was labels: our own units label semi-automatically from `learn_from_own_deploy`,
and opponent units looked like they needed hand labelling. They do not. A
card's sprite is the same sprite whoever plays it, so you can harvest every
card from your *own* deploys (`src/live/harvest.py`) and retint the one
team-coloured part of it (`synth.recolor_team`); and when a real opponent
frame does need naming, the deterministic cycle tracker deduces it from the
kind, the spawn count and what they could afford (`src/live/autolabel.py`).
Neither route asks anyone to label anything.

What remains true of template matching is its bound: a library is specific
to one skin at one resolution, and every new season re-costs it. That is the
reason it is the fallback rather than the target.

Templates are not bundled: they are display- and skin-specific, and shipping
someone else's would be worse than having none. Build them from your own
captures with `learn_from_own_deploy` / `TemplateLibrary.save`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PATCH_SIZE = (24, 24)   # (width, height) templates are normalized to

# Similarity below this is "I don't know" rather than a guess. Feeding a
# wrong identity into the tracker corrupts its cycle model for the rest of
# the match, which is far worse than an unlabelled detection.
MIN_SCORE = 0.55
# ...and a winner that barely beats the runner-up is also "I don't know".
MIN_MARGIN = 0.05
# How much a revealed card outranks an unrevealed one at equal similarity.
PRIOR_BONUS = 0.15


def _normalize(patch: np.ndarray, size=DEFAULT_PATCH_SIZE) -> np.ndarray:
    """Resize to the template grid and zero-mean/unit-norm it.

    Zero-meaning is what makes the score a correlation rather than a
    brightness comparison — the arena's lighting differs between the two
    halves of the board, and a raw dot product would happily match any two
    bright patches.
    """
    if isinstance(patch, Image.Image):
        array = np.asarray(patch.convert("RGB").resize(size, Image.BILINEAR), np.float32)
    else:
        array = np.asarray(patch, np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        array = np.asarray(
            Image.fromarray(array.astype(np.uint8)).resize(size, Image.BILINEAR), np.float32)
    flat = array.reshape(-1) / 255.0
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-9 else flat


@dataclass(frozen=True)
class Identification:
    """Result of one classification attempt.

    `card is None` means "no confident answer" — an explicit outcome, not an
    error. `restricted` records whether the deck prior had narrowed the
    candidate set to the known eight, which is worth logging: accuracy before
    and after the deck reveals is a different number.
    """

    card: str | None
    score: float
    margin: float
    n_candidates: int
    restricted: bool


class TemplateLibrary:
    """Normalized sprite templates, one or more per card."""

    def __init__(self, patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE):
        self.patch_size = patch_size
        self._templates: dict[str, list[np.ndarray]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._templates.values())

    @property
    def cards(self) -> list[str]:
        return sorted(self._templates)

    def add(self, card: str, patch) -> None:
        """Add one example. Several views per card is normal — units are
        animated and face different directions."""
        self._templates.setdefault(card, []).append(_normalize(patch, self.patch_size))

    def similarity(self, card: str, vector: np.ndarray) -> float:
        """Best cosine similarity across this card's examples."""
        return max((float(np.dot(vector, t)) for t in self._templates.get(card, ())),
                   default=-1.0)

    # ------------------------------------------------------------- storage

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "templates.npz",
                 patch_size=np.asarray(self.patch_size),
                 **{card: np.stack(views) for card, views in self._templates.items()})

    @classmethod
    def load(cls, directory: Path | str) -> "TemplateLibrary":
        data = np.load(Path(directory) / "templates.npz")
        size = tuple(int(v) for v in data["patch_size"])
        library = cls(patch_size=(size[0], size[1]))
        for card in data.files:
            if card == "patch_size":
                continue
            library._templates[card] = [np.asarray(v, np.float32) for v in data[card]]
        return library


def extract_patch(
    image,
    detection,
    width: int = 40,
    height: int = 40,
    bar_gap: int = 4,
) -> np.ndarray:
    """Crop the sprite under a `vision.Detection`'s health bar.

    The bar is the reliable anchor (that is why #34 segments it), and the
    unit is drawn directly beneath it, so the patch is defined relative to
    the bar rather than to the detection's projected tile — projection error
    would otherwise smear the crop.
    """
    array = np.asarray(image)
    img_h, img_w = array.shape[:2]
    cx = int(round(detection.blob.center[0]))
    top = detection.blob.y1 + bar_gap
    x0 = max(0, cx - width // 2)
    y0 = max(0, top)
    x1 = min(img_w, x0 + width)
    y1 = min(img_h, y0 + height)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), np.uint8)
    return array[y0:y1, x0:x1, :3]


class DeckPriorClassifier:
    """Template matching narrowed by what the opponent has already shown.

    Pass the same `OpponentTracker` instance the elixir/cycle model uses:
    the revealed-card set is shared state, not a second copy that can drift
    out of agreement with the tracker's cycle.
    """

    def __init__(
        self,
        library: TemplateLibrary,
        tracker=None,
        min_score: float = MIN_SCORE,
        min_margin: float = MIN_MARGIN,
        prior_bonus: float = PRIOR_BONUS,
    ):
        self.library = library
        self.tracker = tracker
        self.min_score = min_score
        self.min_margin = min_margin
        self.prior_bonus = prior_bonus

    def candidates(self) -> tuple[list[str], bool]:
        """(cards to score, whether the deck prior fully restricted them)."""
        known = self.library.cards
        if self.tracker is None:
            return known, False
        revealed = [c for c in self.tracker.candidate_cards() if c in self.library.cards]
        if self.tracker.deck_known and revealed:
            # All eight are out: anything else is a misread by construction.
            return revealed, True
        return known, False

    def classify(self, patch, candidates: list[str] | None = None) -> Identification:
        vector = _normalize(patch, self.library.patch_size)
        restricted = False
        if candidates is None:
            candidates, restricted = self.candidates()
        if not candidates:
            return Identification(None, 0.0, 0.0, 0, restricted)

        prior = set()
        if self.tracker is not None and not restricted:
            # Before the deck is fully revealed, seen cards are merely more
            # likely — a nudge, not a hard filter, because the four unseen
            # cards are still going to show up.
            prior = set(self.tracker.candidate_cards())

        scored = sorted(
            ((self.library.similarity(card, vector)
              + (self.prior_bonus if card in prior else 0.0), card)
             for card in candidates),
            reverse=True)
        best_score, best_card = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_score - runner_up
        if best_score < self.min_score or (len(scored) > 1 and margin < self.min_margin):
            return Identification(None, best_score, margin, len(candidates), restricted)
        return Identification(best_card, best_score, margin, len(candidates), restricted)

    def learn_from_own_deploy(self, image, detection, card: str) -> None:
        """Semi-automatic labelling for our own units.

        We know what we just played, so a friendly detection appearing right
        after our own deploy is a free labelled example. Opponent units get
        no such shortcut — that is where the hand-labelling cost lives.
        """
        self.library.add(card, extract_patch(image, detection))
