"""Conservative live-match loop for a known eight-card deck."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from src.live.config import LiveConfig, Rect
from src.live.device import LiveDevice
from src.live.spells import SpellConfig, SpellWatcher
from src.live.vision import mean_luma, mean_saturation


def scaled_point(config: LiveConfig, point: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    scale_x = image_size[0] / config.reference_size[0]
    scale_y = image_size[1] / config.reference_size[1]
    return (round(point[0] * scale_x), round(point[1] * scale_y))


def scaled_rect(config: LiveConfig, rect: Rect, image_size: tuple[int, int]) -> Rect:
    x, y = scaled_point(config, (rect.x, rect.y), image_size)
    width, height = scaled_point(config, (rect.width, rect.height), image_size)
    return type(rect)(x, y, max(1, width), max(1, height))


@dataclass
class HandCycle:
    hand: list[str]
    next_cards: list[str]

    @classmethod
    def from_config(cls, config: LiveConfig) -> "HandCycle":
        """Seed the cycle from whichever deck the config actually specifies.

        `known_deck` mode calibrates `opening_hand` + `draw_order` explicitly.
        `policy` mode configures `deck` instead, where the listed order *is*
        the cycle: first four in hand, rest queued. Without this fallback the
        cycle would be empty in policy mode and every hand read would be
        blank.
        """
        if config.opening_hand:
            return cls(list(config.opening_hand), list(config.draw_order))
        deck = list(config.deck) or list(config.preset_deck)
        return cls(deck[:4], deck[4:])

    def play(self, slot: int) -> str:
        card = self.hand[slot]
        self.hand[slot] = self.next_cards.pop(0)
        self.next_cards.append(card)
        return card


class LiveMatchRunner:
    """Watch an existing match and deploy configured preset-deck cards.

    The runner does not start matchmaking. It only taps after the calibrated
    match indicator is visible, and only when launched with ``armed=True``.
    """

    def __init__(self, config: LiveConfig, device: LiveDevice, armed: bool = False,
                 log: Callable[[str], None] = print, driver=None):
        self.config = config
        self.device = device
        self.armed = armed
        self.log = log
        self.hand = HandCycle.from_config(config)
        self._was_in_match = False
        self._last_action_at = 0.0
        self._last_logged_choice: tuple[int, str] | None = None
        # None when `homography_anchors` isn't calibrated; the runner then
        # keeps its legacy fixed-target behaviour.
        self.homography = config.homography()
        # A `PolicyDriver` (src/live/bridge.py) when running a checkpoint.
        self.driver = driver
        # Spells are detected across frames, not within one, so the watcher
        # is long-lived state on the runner rather than a per-frame call.
        self.spell_watcher = SpellWatcher(self.homography, config=SpellConfig())
        # Events accumulate between decisions, because the watcher is fed on
        # every capture but drained only when a decision is made.
        self._spell_events: list = []
        self._warned_capture_rate = False
        self._match_started_at = 0.0
        self._degraded = False

    # --------------------------------------------------------- perception

    def perceive(self, image) -> "object | None":
        """Assemble a `LiveObservation` from one frame, or None if the frame
        cannot be read well enough to act on.

        Returning None is deliberate: a fabricated observation is worse than
        no decision. If own elixir cannot be read, every affordability mask
        downstream is invented, and the policy would confidently play cards
        that are not available.
        """
        from src.live.bridge import LiveObservation, PerceivedSpell, PerceivedUnit
        from src.live.vision import TEAM_HOSTILE, detect_units, read_elixir

        config = self.config
        if config.elixir_bar is None or self.homography is None:
            return None
        elixir = read_elixir(image, scaled_rect(config, config.elixir_bar, image.size))
        if elixir is None:
            return None

        homography = self.homography.scaled_to(config.reference_size, image.size)
        units = []
        try:
            from src.simulator.cards import load_arena
            for d in detect_units(image, homography, arena=load_arena()):
                units.append(PerceivedUnit(
                    card="", tile_x=d.tile_x, tile_y=d.tile_y,
                    hostile=d.team == TEAM_HOSTILE,
                    hp_fraction=d.hp_fraction, hp_confident=d.hp_confident))
        except ValueError:
            # A degenerate projection is recoverable — act on hand and elixir
            # alone this frame rather than dropping the whole decision.
            units = []

        return LiveObservation(
            hand=list(self.hand.hand),
            next_card=self.hand.next_cards[0] if self.hand.next_cards else "",
            own_elixir=elixir,
            match_time=max(0.0, time.monotonic() - self._match_started_at),
            units=units,
            spells=self._drain_spells(),
        )

    def _observe_spells(self, image) -> None:
        """Feed one capture to the spell watcher and buffer what it emits.

        Separate from `perceive` because the two run on different clocks.
        Perception runs when a decision is due; the watcher needs *every*
        frame, since a bloom is only visible as a difference between
        consecutive ones.

        A capture rate too slow to resolve a bloom does not degrade
        detection gracefully, it stops it, so the watcher is asked whether
        it is starved and the answer is logged once rather than left to look
        like a quiet match.
        """
        if self.homography is None:
            return
        from src.simulator.cards import load_arena

        self.spell_watcher.arena = load_arena()
        try:
            homography = self.homography.scaled_to(
                self.config.reference_size, image.size)
            self._spell_events.extend(self.spell_watcher.observe(
                image, now=time.monotonic(), homography=homography))
        except ValueError:
            return

        if self.spell_watcher.starved and not self._warned_capture_rate:
            self._warned_capture_rate = True
            self.log(
                f"Captures are {self.spell_watcher.frame_interval:.2f}s apart; "
                f"spell detection needs "
                f"{self.spell_watcher.config.max_frame_interval:.2f}s or less. "
                "Lower `poll_seconds`, or expect no spells to be reported.")

    def _drain_spells(self) -> list:
        """Spell casts buffered since the last decision.

        Identity is resolved against the full spell roster, which means it
        usually abstains — several spells share a footprint, and radius
        alone cannot separate them. That is the honest answer until a
        deterministic `OpponentTracker` is threaded through to supply
        `candidates` and `elixir_drop`; with those the same call names most
        casts, and it needs no labelled data to do it. An abstaining spell
        still carries a measured footprint, which is most of what channels
        6/7 encode.
        """
        from src.live.bridge import PerceivedSpell
        from src.live.spells import resolve_identity
        from src.simulator.cards import load_cards

        events, self._spell_events = self._spell_events, []
        if not events:
            return []

        cards = load_cards()
        return [PerceivedSpell(
            card=resolve_identity(event, cards),
            tile_x=event.tile_x, tile_y=event.tile_y,
            # Perception cannot tell whose spell it is from the VFX — spells
            # are not team-tinted the way health bars are. Assuming hostile
            # is the safe default: a spell of ours read as theirs costs the
            # policy some caution, while one of theirs read as ours would
            # have it ignore an area that is about to be cleared.
            hostile=True,
            radius=event.radius,
            confidence=event.confidence,
        ) for event in events]

    def tile_to_pixel(self, tile: tuple[float, float],
                      image_size: tuple[int, int]) -> tuple[int, int] | None:
        """Arena tile -> tap point on this capture, or None if uncalibrated.

        This is the piece that lets a *policy* drive live play instead of the
        heuristic below: the policy picks a placement cell, and this turns it
        into a tap. The arena is drawn in perspective, so it goes through the
        calibrated homography rather than `scaled_point`, which is only
        correct for the flat HUD.
        """
        if self.homography is None:
            return None
        matrix = self.homography.scaled_to(self.config.reference_size, image_size)
        px, py = matrix.tile_to_pixel(*tile)
        return round(px), round(py)

    def run_forever(self, max_consecutive_errors: int = 10) -> None:
        """Poll until interrupted, surviving transient capture failures.

        A watcher runs unattended for a whole match, so a single recoverable
        hiccup — the window momentarily minimized (empty client area), a
        blocked screen grab, a dropped touch injection — must not end the
        session. Persistent failure still stops rather than spinning forever,
        because that means the setup is genuinely broken (game closed, window
        title changed) and retrying can't fix it.
        """
        consecutive_errors = 0
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                self.log("Interrupted; stopping.")
                raise
            except (RuntimeError, OSError, ValueError) as error:
                consecutive_errors += 1
                self.log(
                    f"Recoverable error ({consecutive_errors}/{max_consecutive_errors}): {error}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Giving up after {max_consecutive_errors} consecutive errors; "
                        f"last was: {error}"
                    ) from error
            else:
                consecutive_errors = 0
            time.sleep(self.config.poll_seconds)

    def step(self) -> None:
        image = self.device.screenshot()
        in_match = mean_saturation(image, scaled_rect(self.config, self.config.match_indicator, image.size)) >= self.config.match_min_saturation
        if in_match and not self._was_in_match:
            self.hand = HandCycle.from_config(self.config)
            self._last_logged_choice = None
            self._match_started_at = time.monotonic()
            self._degraded = False
            # A track carried across a match boundary would emit its spell
            # against the new board, at a tile that means nothing there.
            self.spell_watcher.reset()
            self._spell_events = []
            self.log("Match detected; hand cycle reset.")
        self._was_in_match = in_match

        # Feed the spell watcher here, deliberately ahead of the cooldown
        # return below. A spell blooms and collapses in well under a second
        # while `action_cooldown_seconds` defaults to 1.2, so sampling only
        # the frames we act on would miss every cast in the match.
        if in_match:
            self._observe_spells(image)

        if not in_match or time.monotonic() - self._last_action_at < self.config.action_cooldown_seconds:
            return

        if self.driver is not None and self._step_policy(image):
            return

        ready_slots = [
            slot for slot, region in enumerate(self.config.card_ready_regions)
            if mean_luma(image, scaled_rect(self.config, region, image.size)) >= self.config.card_ready_min_luma
        ]
        choice = self._choose_action(ready_slots)
        if choice is None:
            return
        slot, card = choice
        if not self.armed and choice == self._last_logged_choice:
            return
        if self.config.decision_mode == "dynamic_slots":
            target = self.config.dynamic_target
        else:
            target = self.config.placements.get(card)
            if target is None:
                return
        card_slot = scaled_point(self.config, self.config.card_slots[slot], image.size)
        target = scaled_point(self.config, target, image.size)
        self.log(f"{'Playing' if self.armed else 'Would play'} {card} from slot {slot + 1} at {target}.")
        if self.armed:
            self.device.tap(*card_slot)
            time.sleep(self.config.tap_delay_seconds)
            self.device.tap(*target)
            if self.config.decision_mode == "known_deck":
                self.hand.play(slot)
        else:
            self._last_logged_choice = choice
        self._last_action_at = time.monotonic()

    def _step_policy(self, image) -> bool:
        """One policy-driven decision. Returns False to fall through to the
        heuristic.

        Degrading rather than failing is the whole reason the heuristic is
        kept: if vision drops out mid-match — window occluded, an arena skin
        the thresholds do not handle — tapping a safe card at a safe spot is
        a far better failure mode than feeding the policy a fabricated board
        and letting it act on it.
        """
        observation = self.perceive(image)
        if observation is None:
            if not self._degraded:
                self._degraded = True
                self.log("Perception unavailable; falling back to the heuristic.")
            return False
        if self._degraded:
            self._degraded = False
            self.log("Perception recovered; policy driving again.")

        action = self.driver.decide(observation)
        if action is None:
            return True   # the policy chose to hold; that is a real decision

        target = self.tile_to_pixel(action.tile, image.size)
        if target is None:
            return False
        card_slot = scaled_point(self.config, self.config.card_slots[action.slot], image.size)
        self.log(f"{'Playing' if self.armed else 'Would play'} {action.card} "
                 f"from slot {action.slot + 1} at tile "
                 f"({action.tile[0]:.1f}, {action.tile[1]:.1f}) -> {target}.")
        if self.armed:
            self.device.tap(*card_slot)
            time.sleep(self.config.tap_delay_seconds)
            self.device.tap(*target)
            # Advance the deterministic cycle only when the tap actually went
            # out; advancing on a dry run would desynchronise the hand from
            # the game for the rest of the match.
            self.hand.play(action.slot)
        self._last_action_at = time.monotonic()
        return True

    def _choose_action(self, ready_slots: list[int]) -> tuple[int, str] | None:
        if self.config.decision_mode == "dynamic_slots":
            for slot in self.config.slot_priority:
                if slot in ready_slots:
                    return slot, f"visible slot {slot + 1}"
            return None
        for card in self.config.priority:
            for slot in ready_slots:
                if self.hand.hand[slot] == card and card in self.config.placements:
                    return slot, card
        return None
