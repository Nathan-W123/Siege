"""Drive a trained policy from live perception.

The problem this solves
-----------------------
`PolicyBot.decide()` and `masking.build_action_masks()` both take a
`BattleEngine`. Live play has perception — detections, an elixir reading, a
hand cycle — and no engine. That mismatch is the only structural reason the
trained policy could not drive live play; everything else was already in
place.

The approach: **reconstruct a shadow engine** from what was observed, then
reuse the existing code paths verbatim. The alternative — building the
observation dict and action masks directly from perception — means
reimplementing deploy legality, elixir affordability and the placement grid
on the live side, where they would drift out of agreement with the simulator
the policy was trained against. Reusing `build_action_masks` means the
policy sees exactly the action space it learned.

What is real and what is not
----------------------------
The shadow engine is faithful about what perception can see and *silent*
about what it cannot:

- **Real:** own hand and elixir, tower alive/dead, enemy and friendly unit
  positions, unit identity where the classifier is confident.
- **Approximated:** unit HP (from health-bar fill; full bars are
  indistinguishable from occluded ones), and unit identity where the
  classifier abstains.
- **Absent:** attack cooldowns, targeting locks, deploy timers, building
  self-destruct timers, charge and ramp state. The policy never observes
  these — they are not in any observation tier — so leaving them at defaults
  costs nothing.

The one thing that would be dishonest is inventing opponent elixir. It comes
from `OpponentTracker`, which derives it from observed play, and it is only
consumed when running a `full`-tier policy (which should not be used live at
all; `check_tier` rejects it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agent import masking, obs_layout
from src.agent.network import PolicyNetwork, masks_to_tensors, obs_to_tensors
from src.simulator.cards import ArenaConfig, CardStats
from src.simulator.constants import HAND_SIZE, PLACE_COLS, CardType, Side
from src.simulator.engine import BattleEngine
from src.simulator.entities import PendingSpell, Unit

# Perceived units carry no deploy timer, so they are back-dated far enough
# that the engine treats them as fully active. A live detection is by
# definition a unit already on the arena.
_LONG_AGO = -1000.0


@dataclass
class PerceivedUnit:
    """One entity on the arena the vision pipeline reported.

    `kind` is carried separately from `card` on purpose. Identity and kind
    are *different detection problems* with very different difficulty: which
    of eight cards a sprite is, is hard; whether it is moving or bolted to a
    tile is easy, and a detector answers it far more often than it answers
    the first. Keeping them apart lets the easy answer through when the hard
    one abstains, which is exactly the case `ShadowEngine._stand_in` exists
    to handle.

    When `card` names a card in the table, that card's own type wins — the
    table is authoritative and perception's guess at kind is not needed.
    """

    card: str            # "" when the classifier abstained
    tile_x: float
    tile_y: float
    hostile: bool
    hp_fraction: float = 1.0
    hp_confident: bool = False
    kind: CardType = CardType.TROOP


@dataclass
class PerceivedSpell:
    """One spell cast the frame-difference detector reported.

    Spells are *events*, not entities: a fireball is a half-second of
    expanding VFX with no health bar and nothing left behind, so the
    health-bar segmenter that finds troops cannot see one at any threshold.
    They arrive on their own channel for that reason (`src.live.spells`).

    What is measured and what is not, kept apart deliberately:

    - **Measured:** position and `radius`, straight off the VFX footprint.
    - **Not visible at all:** damage. No number of pixels reveals it; it is
      looked up from the card table once identity is resolved, and taken
      from a neutral stand-in when it is not.

    So an unidentified spell still reports a truthful footprint with a
    guessed magnitude — which is the right way round, because channels 6/7
    are dominated by *where* the area denial is.
    """

    card: str            # "" when identity could not be resolved
    tile_x: float
    tile_y: float
    hostile: bool
    radius: float = 0.0          # tiles, measured; 0 = fall back to the card
    time_to_impact: float = 0.0  # seconds until it lands
    confidence: float = 0.0


@dataclass
class LiveObservation:
    """Everything perception managed to establish about the current frame."""

    hand: list[str]
    next_card: str
    own_elixir: float
    match_time: float = 0.0
    units: list[PerceivedUnit] = field(default_factory=list)
    spells: list[PerceivedSpell] = field(default_factory=list)
    # Tower state, keyed as the engine names them.
    own_left_alive: bool = True
    own_right_alive: bool = True
    enemy_left_alive: bool = True
    enemy_right_alive: bool = True
    # Only read by a `full`-tier policy, which is rejected for live play.
    opponent_elixir: float = 0.0


class ShadowEngine:
    """Builds a `BattleEngine` that mirrors what was perceived.

    A fresh engine per frame rather than one mutated in place: the live
    bridge has no reliable way to track unit identity across frames, so
    carrying stale units forward would accumulate ghosts. Rebuilding is also
    cheap next to the policy forward pass.
    """

    def __init__(self, cards: dict[str, CardStats], arena: ArenaConfig,
                 deck: list[str], fallback_card: str = "knight",
                 fallback_building: str = "cannon",
                 fallback_spell: str = "fireball"):
        missing = [c for c in deck if c not in cards]
        if missing:
            raise ValueError(f"deck cards not in the card table: {missing}")
        if len(deck) < HAND_SIZE + 1:
            raise ValueError(f"deck needs at least {HAND_SIZE + 1} cards, got {len(deck)}")
        self.cards = cards
        self.arena = arena
        self.deck = list(deck)
        # Stand-ins for a detection the classifier could not name. Something
        # from the deck would be a worse lie — it would imply knowledge of
        # the opponent's cards that was never observed.
        #
        # One stand-in per *kind*, not one overall. See `_stand_in`: getting
        # the kind wrong is a different and worse error than getting the card
        # wrong, so the abstention path must at least preserve kind.
        self.fallback_card = fallback_card if fallback_card in cards else self.deck[0]
        self.fallback_building = self._pick_stand_in(fallback_building, CardType.BUILDING)
        self.fallback_spell = self._pick_stand_in(fallback_spell, CardType.SPELL)

    def _pick_stand_in(self, preferred: str, kind: CardType) -> str | None:
        """Name a card of `kind` to stand in for unnamed detections of it.

        Falls back to any card of the right kind rather than raising: a card
        table without, say, a cannon is a perfectly valid table, and losing
        the stand-in would silently take the whole kind out of the
        observation. None means the table has no card of this kind at all,
        which `_stand_in` handles by dropping the detection.
        """
        entry = self.cards.get(preferred)
        if entry is not None and entry.type == kind:
            return preferred
        return next((n for n, c in sorted(self.cards.items()) if c.type == kind), None)

    def build(self, observation: LiveObservation) -> BattleEngine:
        deck_cards = [self.cards[c] for c in self.deck]
        engine = BattleEngine(deck_cards, list(deck_cards), self.arena,
                              seed=0, cards=self.cards)
        engine.time = float(observation.match_time)

        self._apply_hand(engine, observation)
        self._apply_towers(engine, observation)
        for unit in observation.units:
            self._spawn(engine, unit)
        for spell in observation.spells:
            self._cast(engine, spell)
        return engine

    # ---------------------------------------------------------------- parts

    def _apply_hand(self, engine: BattleEngine, observation: LiveObservation) -> None:
        """Pin the hand to the deterministic cycle and the elixir to the bar.

        The hand is *known*, not guessed: `HandCycle` simulates the 8-card
        cycle from the configured deck, which is why live play never needed
        to recognize its own cards.
        """
        me = engine.players[Side.BOTTOM]
        hand = [c for c in observation.hand if c in self.cards][:HAND_SIZE]
        while len(hand) < HAND_SIZE:
            hand.append(self.deck[len(hand)])
        me.hand = [self.cards[c] for c in hand]
        rest = [c for c in self.deck if c not in hand]
        nxt = observation.next_card if observation.next_card in self.cards else None
        if nxt and nxt in rest:
            rest = [nxt] + [c for c in rest if c != nxt]
        me._queue = [self.cards[c] for c in rest] or [self.cards[self.deck[0]]]
        me.elixir = max(0.0, min(float(observation.own_elixir), engine.arena.elixir_max))
        engine.players[Side.TOP].elixir = max(
            0.0, min(float(observation.opponent_elixir), engine.arena.elixir_max))

    def _apply_towers(self, engine: BattleEngine, observation: LiveObservation) -> None:
        alive = {
            (Side.BOTTOM, "princess_left"): observation.own_left_alive,
            (Side.BOTTOM, "princess_right"): observation.own_right_alive,
            (Side.TOP, "princess_left"): observation.enemy_left_alive,
            (Side.TOP, "princess_right"): observation.enemy_right_alive,
        }
        for tower in engine.towers:
            if alive.get((tower.side, tower.kind), True):
                continue
            tower.hp = 0.0
            # A fallen princess activates that side's king, and it opens the
            # pocket the placement mask depends on.
            engine._king_of(tower.side).activated = True

    def _stand_in(self, perceived: PerceivedUnit) -> CardStats | None:
        """The card table entry to use for a detection, named or not.

        Kind survives abstention even when identity does not, and that
        asymmetry is the whole point. `obs_layout.encode_spatial` routes HP
        by `Unit.is_building`: channels 2/3 are building HP, 0/1 are troop
        HP. So a Cannon standing in as a Knight is not one mislabelled
        entity — it moves that HP out of the enemy-building channel into the
        enemy-troop channel, and tells the policy a structure bolted to a
        tile is walking at its tower. Wrong card, right kind, costs the
        policy some stat precision; wrong kind costs it the read.

        Returns None only when the table has no card of the perceived kind,
        in which case the caller drops the detection rather than inventing a
        kind for it.
        """
        named = self.cards.get(perceived.card)
        if named is not None:
            return named          # the table is authoritative about kind
        if perceived.kind == CardType.BUILDING:
            return self.cards.get(self.fallback_building or "")
        return self.cards.get(self.fallback_card)

    def _spawn(self, engine: BattleEngine, perceived: PerceivedUnit) -> None:
        stats = self._stand_in(perceived)
        if stats is None or stats.type == CardType.SPELL:
            # A spell has no unit to spawn. Reaching here means perception
            # put a spell in `units`; drop it rather than materialising a
            # zero-HP troop that would sit on the arena forever.
            return
        side = Side.TOP if perceived.hostile else Side.BOTTOM
        hp = max(1.0, stats.hp * max(0.0, min(perceived.hp_fraction, 1.0)))
        unit = Unit(
            id=engine._new_id(), stats=stats, side=side,
            x=min(max(perceived.tile_x, 0.0), engine.arena.width - 0.01),
            y=min(max(perceived.tile_y, 0.0), engine.arena.height - 0.01),
            hp=hp, cooldown=stats.hit_speed,
            elixir_value=stats.cost / max(stats.count, 1),
            deployed_at=_LONG_AGO,
        )
        engine.units.append(unit)
        engine._by_id[unit.id] = unit

    def _cast(self, engine: BattleEngine, perceived: PerceivedSpell) -> None:
        """Put a perceived spell into the engine's pending list.

        The shadow engine is never stepped — it is built to be *read* by
        `encode_obs` and `build_action_masks` — so a pending spell here is
        purely an observation carrier and never applies damage. `resolve_at`
        is still set honestly from the measured time to impact, so that
        stepping one (in a test, or a future rollout) does the right thing.
        """
        stats = self.cards.get(perceived.card) or self.cards.get(self.fallback_spell or "")
        if stats is None or stats.type != CardType.SPELL:
            return
        # Radius is the one spell property that is genuinely visible, so a
        # measurement beats the table; damage is never visible, so the table
        # (or the stand-in) is all there is. Mixing the two is not sloppiness
        # — it is using each source where it is actually informed.
        radius = perceived.radius if perceived.radius > 0 else stats.spell_radius
        engine.spells.append(PendingSpell(
            side=Side.TOP if perceived.hostile else Side.BOTTOM,
            x=min(max(perceived.tile_x, 0.0), engine.arena.width - 0.01),
            y=min(max(perceived.tile_y, 0.0), engine.arena.height - 0.01),
            radius=radius,
            damage=stats.spell_damage,
            tower_multiplier=stats.tower_multiplier,
            resolve_at=engine.time + max(0.0, perceived.time_to_impact),
            card_name=perceived.card or stats.name,
        ))


@dataclass(frozen=True)
class LiveAction:
    """A decision, in the terms the tapping layer needs."""

    slot: int            # 0-3, index into the visible card slots
    card: str
    tile: tuple[float, float]


def check_tier(net: PolicyNetwork) -> None:
    """Refuse to run a policy that reads what a player cannot see."""
    tier = obs_layout.resolve_tier(net.config.tier)
    if tier == obs_layout.TIER_FULL:
        raise ValueError(
            "this checkpoint is `full` tier — it reads the opponent's exact "
            "elixir, which no player can see, so it is simulator-only. Train "
            "or distil a `human`-tier policy for live play "
            "(configs/training_human.yaml).")


class PolicyDriver:
    """Turns a live observation into a card + placement using a checkpoint."""

    def __init__(
        self,
        net: PolicyNetwork,
        card_names: list[str],
        cards: dict[str, CardStats],
        arena: ArenaConfig,
        deck: list[str],
        deterministic: bool = True,
    ):
        check_tier(net)
        self.net = net
        self.card_to_id = {n: i for i, n in enumerate(card_names)}
        self.shadow = ShadowEngine(cards, arena, deck)
        self.deterministic = deterministic
        self.last_engine: BattleEngine | None = None

    def decide(self, observation: LiveObservation) -> LiveAction | None:
        """None means "play nothing this frame" — a real choice, not a failure."""
        import torch

        engine = self.shadow.build(observation)
        self.last_engine = engine
        device = next(self.net.parameters()).device

        obs = obs_layout.encode_obs(
            engine, Side.BOTTOM, self.card_to_id, self.net.config.tier,
            with_units=getattr(self.net.config, "use_set_encoder", False))
        m = masking.build_action_masks(engine, Side.BOTTOM)
        masks = {"card": _with_noop(m["card"], m["place"]), "place": m["place"]}

        with torch.no_grad():
            actions, _, _ = self.net.act(obs_to_tensors(obs, device),
                                         masks_to_tensors(masks, device),
                                         deterministic=self.deterministic)
        choice, cell = int(actions[0, 0]), int(actions[0, 1])
        if choice == 0:
            return None
        slot = choice - 1
        row, col = divmod(cell, PLACE_COLS)
        tile = masking.cell_to_xy(Side.BOTTOM, col, row, engine.arena.height)
        return LiveAction(slot=slot, card=engine.players[Side.BOTTOM].hand[slot].name,
                          tile=tile)


def _with_noop(card_mask, place_mask):
    """Prepend the no-op choice, matching what the env feeds the policy."""
    import numpy as np

    return np.concatenate(([True], card_mask & place_mask.any(axis=1)))
