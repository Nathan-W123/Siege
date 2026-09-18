"""A checkpoint records the deck it trained on, and the bridge checks it.

The deck is a *runtime* argument — `PolicyDriver` takes it, and every
checkpoint carries an embedding slot for all cards — so handing a policy a
deck it has never played is legal and silent. It does not error; it just
plays badly, in a way indistinguishable from the policy being bad. Card
levels were already recorded for this exact reason; the deck was not.
"""
from __future__ import annotations

import torch

from src.agent.network import make_network
from src.agent.selfplay import checkpoint_deck, save_checkpoint
from src.live.__main__ import _warn_on_deck_mismatch

N_CARDS = 12
CARD_NAMES = [f"card_{i}" for i in range(N_CARDS)]
SMALL = {"conv_channels": (4,), "cnn_out": 16, "fusion_mlp": 32}
DECK = ["knight", "archers", "goblins", "giant",
        "musketeer", "minions", "fireball", "cannon"]


def _checkpoint(tmp_path, deck=None, name="net.pt"):
    net = make_network(N_CARDS, SMALL)
    path = tmp_path / name
    save_checkpoint(net, CARD_NAMES, path, deck=deck)
    return path


# ------------------------------------------------------------- provenance


def test_a_pinned_deck_is_recorded(tmp_path):
    assert checkpoint_deck(_checkpoint(tmp_path, deck=DECK)) == DECK


def test_a_pooled_stage_records_no_deck(tmp_path):
    """None is a real answer: a `full_pool` policy resamples its deck every
    episode, so there is no single deck it could claim to have trained on."""
    assert checkpoint_deck(_checkpoint(tmp_path)) is None


def test_older_checkpoints_still_load(tmp_path):
    """Every checkpoint already on disk predates this. None of them may
    break, and none of them may claim a deck they never recorded."""
    net = make_network(N_CARDS, SMALL)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": net.state_dict(),
                "config": {"n_cards": N_CARDS, **SMALL},
                "card_names": CARD_NAMES}, path)

    assert checkpoint_deck(path) is None


def test_pinned_deck_resolves_only_for_a_pinned_stage():
    from src.agent.train import pinned_deck
    from src.decks.catalog import DeckCatalog
    from src.training.curriculum import load_curriculum

    catalog = DeckCatalog()
    stages = {s.name: s for s in load_curriculum()}

    assert pinned_deck(stages["my_deck"], catalog) == catalog.resolve("my_deck")
    assert pinned_deck(stages["full_pool"], catalog) is None


# ------------------------------------------------------------- the warning


def _messages(trained, deck):
    out = []
    _warn_on_deck_mismatch(trained, deck, out.append)
    return out


def test_a_matching_deck_says_nothing():
    assert _messages(DECK, DECK) == []


def test_order_alone_is_not_a_mismatch():
    """The live config lists cards in cycle order and the catalog does not.
    Warning on that would train people to ignore the warning."""
    assert _messages(DECK, list(reversed(DECK))) == []


def test_a_different_deck_warns_and_names_the_unseen_cards():
    swapped = DECK[:-1] + ["mega_knight"]

    messages = _messages(DECK, swapped)

    assert len(messages) == 1
    assert "WARNING" in messages[0]
    assert "mega_knight" in messages[0]
    assert "my_deck" in messages[0], "the warning should say what to do"


def test_it_warns_rather_than_refusing():
    """Handing a pooled policy an arbitrary deck is legal and sometimes the
    whole point, so this cannot be an error — only something you are told."""
    messages = _messages(DECK, ["mega_knight"] * 8)

    assert messages and "will run" in messages[0]


def test_an_unrecorded_deck_is_a_neutral_note_not_an_alarm():
    """A checkpoint with no deck is either pooled or old, and those are
    indistinguishable from here, so neither gets a false alarm."""
    messages = _messages(None, DECK)

    assert len(messages) == 1
    assert "WARNING" not in messages[0]
    assert "cannot check" in messages[0]
