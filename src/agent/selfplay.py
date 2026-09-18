"""Adapters that let a policy network play anywhere a Bot can, and vice versa.

- PolicyBot: implements the src.bots.base.Bot protocol (decide(engine, side)),
  so trained checkpoints plug into match_runner, benchmark, and the league.
- BotOpponent / PolicyOpponent: implement the env Opponent protocol
  (act(env, side)) for opponents living inside CRBattleEnv during training.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.agent import masking, obs_layout
from src.agent.network import PolicyNetwork, make_network, masks_to_tensors, obs_to_tensors
from src.bots.base import Action, Bot
from src.simulator.cards import load_cards
from src.simulator.constants import PLACE_COLS, Side
from src.simulator.engine import BattleEngine


def save_checkpoint(net: PolicyNetwork, card_names: list[str], path: Path,
                    card_levels=None, deck=None) -> None:
    """Persist a policy.

    `card_levels` is recorded as provenance, not as something the network
    needs: a policy trained against level-13 stats has learned level-13
    breakpoints, and running it live against a level-9 collection would
    quietly mean a different game. Storing it lets the live bridge check.

    `deck` is recorded for exactly the same reason, and it had been missing.
    The deck is a *runtime* argument — `PolicyDriver` takes it, and every
    checkpoint carries an embedding slot for all 171 cards — so nothing
    stops a policy being handed a deck it has never played. It will not
    error; it will just be bad, in a way that looks like the policy being
    bad rather than like a configuration mistake. Recording what it trained
    on lets the bridge say so.

    None means the agent's deck was sampled from a pool rather than pinned,
    which is a real answer: a `full_pool` policy has no single deck to
    compare against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "config": asdict(net.config),
        "card_names": list(card_names),
    }
    if deck is not None:
        payload["deck"] = list(deck)
    if card_levels is not None:
        payload["card_levels"] = (card_levels.to_dict()
                                  if hasattr(card_levels, "to_dict") else dict(card_levels))
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[PolicyNetwork, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    n_cards = cfg.pop("n_cards")
    cfg["conv_channels"] = tuple(cfg["conv_channels"])
    net = make_network(n_cards, cfg)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["card_names"]


def checkpoint_deck(path: Path) -> list[str] | None:
    """The deck a checkpoint was trained on, or None.

    None covers both "trained on a sampled pool" and "predates this being
    recorded", and the caller cannot tell them apart. That is why the live
    bridge warns on a mismatch rather than refusing: it cannot prove the
    policy is wrong for this deck, only that it cannot vouch for it.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    deck = ckpt.get("deck")
    return list(deck) if deck else None


def checkpoint_card_levels(path: Path):
    """The `CardLevels` a checkpoint was trained under, or None if it predates
    the feature (in which case it is level 1 — what `cards.yaml` holds)."""
    from src.simulator.levels import CardLevels

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = ckpt.get("card_levels")
    return CardLevels.from_dict(raw) if raw else CardLevels()


def _obs_and_masks(
    net: PolicyNetwork,
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Numpy observation + action masks for one engine/seat."""
    obs = obs_layout.encode_obs(engine, side, card_to_id, net.config.tier,
                                with_units=getattr(net.config, "use_set_encoder", False))
    m = masking.build_action_masks(engine, side)
    card = np.concatenate(([True], m["card"] & m["place"].any(axis=1)))
    return obs, {"card": card, "place": m["place"]}


def policy_action(
    net: PolicyNetwork,
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
    deterministic: bool = False,
) -> tuple[int, int]:
    """(card_choice, cell) for the given engine state."""
    device = next(net.parameters()).device
    obs, masks = _obs_and_masks(net, engine, side, card_to_id)
    actions, _, _ = net.act(obs_to_tensors(obs, device),
                            masks_to_tensors(masks, device),
                            deterministic=deterministic)
    return int(actions[0, 0]), int(actions[0, 1])


def policy_actions_batched(
    net: PolicyNetwork,
    engines: list[BattleEngine],
    side: Side,
    card_to_id: dict[str, int],
    deterministic: bool = False,
) -> np.ndarray:
    """One forward pass covering many engines that share a policy.

    Opponent policies were previously evaluated one env at a time, so a
    vectorized step issued N batch-of-1 forwards. Those are dominated by
    per-call overhead rather than arithmetic — profiling put ~90% of training
    wall-clock in torch, not the simulator — so collapsing them into a single
    batched call is close to free throughput.

    Returns an ``(N, 2)`` array of ``(card_choice, cell)`` rows aligned with
    ``engines``.
    """
    if not engines:
        return np.zeros((0, 2), dtype=np.int64)
    device = next(net.parameters()).device
    pairs = [_obs_and_masks(net, e, side, card_to_id) for e in engines]
    obs = {k: np.stack([o[k] for o, _ in pairs]) for k in pairs[0][0]}
    masks = {k: np.stack([m[k] for _, m in pairs]) for k in pairs[0][1]}
    actions, _, _ = net.act(obs_to_tensors(obs, device),
                            masks_to_tensors(masks, device),
                            deterministic=deterministic)
    return actions.cpu().numpy()


class PolicyBot:
    """A trained policy exposed through the scripted-bot interface."""

    def __init__(self, net: PolicyNetwork, card_names: list[str],
                 name: str = "policy", deterministic: bool = False):
        self.net = net
        self.card_to_id = {n: i for i, n in enumerate(card_names)}
        self.name = name
        self.deterministic = deterministic
        # Recurrent policies need their memory carried between decisions.
        # Keyed by engine identity so one bot can drive several matches
        # (benchmark, league, both env seats) without their memories mixing.
        self._hidden: dict[int, object] = {}
        self._last_time: dict[int, float] = {}

    @classmethod
    def load(cls, path: Path, name: str | None = None,
             deterministic: bool = False) -> "PolicyBot":
        net, card_names = load_checkpoint(Path(path))
        return cls(net, card_names, name=name or Path(path).stem,
                   deterministic=deterministic)

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        if self.net.config.use_recurrence:
            return self.decode_row(engine, side,
                                   self._recurrent_row(engine, side))
        choice, cell = policy_action(self.net, engine, side, self.card_to_id,
                                     deterministic=self.deterministic)
        return self.decode_row(engine, side, (choice, cell))

    def _recurrent_row(self, engine: BattleEngine, side: Side):
        """One recurrent decision, carrying this match's hidden state.

        Without this a recurrent checkpoint would be evaluated as if it were
        memoryless — every benchmark number would silently understate it.

        Memory is reset when the engine's clock goes backwards, which is how
        a reused key signals a fresh match; engine objects are recreated per
        match, so in practice the key is simply new.
        """
        import torch

        device = next(self.net.parameters()).device
        key = (id(engine), int(side))
        hidden = self._hidden.get(key)
        if hidden is None or engine.time < self._last_time.get(key, -1.0):
            hidden = self.net.initial_hidden(1, device)
        self._last_time[key] = engine.time

        obs, masks = _obs_and_masks(self.net, engine, side, self.card_to_id)
        actions, _, _, hidden = self.net.act_recurrent(
            obs_to_tensors(obs, device), masks_to_tensors(masks, device),
            hidden, None, deterministic=self.deterministic)
        self._hidden[key] = hidden
        return int(actions[0, 0]), int(actions[0, 1])

    def decode_row(self, engine: BattleEngine, side: Side,
                   row) -> Action | None:
        """Turn a ``(card_choice, cell)`` pair into an Action.

        Split out of `decide` so a caller that already ran the network in a
        batch (see `policy_actions_batched`) can reuse the same decoding
        rather than re-running inference per env.
        """
        choice, cell = int(row[0]), int(row[1])
        if choice == 0:
            return None
        grid_row, col = divmod(cell, PLACE_COLS)
        x, y = masking.cell_to_xy(side, col, grid_row, engine.arena.height)
        return Action(choice - 1, x, y)

    # Also usable directly as an env opponent.
    def act(self, env, side: Side) -> tuple[int, float, float] | None:
        action = self.decide(env.engine, side)
        return None if action is None else (action.slot, action.x, action.y)

    def batch_key(self):
        """Envs whose opponents share a key can be evaluated in one forward."""
        return (id(self.net), self.deterministic)

    def batched_rows(self, engines, side: Side):
        """Actions for many engines in one forward, memory included.

        Recurrent policies cannot use the plain `policy_actions_batched`
        path — it has no hidden state to thread, so it would play them as if
        they had none. Rather than give up the ~7x batching win, each
        engine's memory is gathered, stacked into a single sequence step, and
        scattered back afterwards.
        """
        if not self.net.config.use_recurrence:
            return policy_actions_batched(self.net, engines, side, self.card_to_id,
                                          deterministic=self.deterministic)
        device = next(self.net.parameters()).device
        keys = [(id(e), int(side)) for e in engines]
        hiddens = []
        for engine, key in zip(engines, keys):
            h = self._hidden.get(key)
            if h is None or engine.time < self._last_time.get(key, -1.0):
                h = self.net.initial_hidden(1, device)
            hiddens.append(h)
            self._last_time[key] = engine.time
        hidden = torch.cat(hiddens, dim=1)

        pairs = [_obs_and_masks(self.net, e, side, self.card_to_id) for e in engines]
        obs = {k: np.stack([o[k] for o, _ in pairs]) for k in pairs[0][0]}
        masks = {k: np.stack([m[k] for _, m in pairs]) for k in pairs[0][1]}
        actions, _, _, hidden = self.net.act_recurrent(
            obs_to_tensors(obs, device), masks_to_tensors(masks, device),
            hidden, None, deterministic=self.deterministic)
        for i, key in enumerate(keys):
            self._hidden[key] = hidden[:, i:i + 1].contiguous()
        return actions.cpu().numpy()


class BotOpponent:
    """Wrap a scripted Bot as an env Opponent."""

    def __init__(self, bot: Bot):
        self.bot = bot
        self.name = bot.name

    def act(self, env, side: Side) -> tuple[int, float, float] | None:
        action = self.bot.decide(env.engine, side)
        return None if action is None else (action.slot, action.x, action.y)


def default_card_names() -> list[str]:
    return list(load_cards().keys())
