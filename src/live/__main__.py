"""Run the calibrated live-play bridge."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.live.config import LiveConfig, load_live_config
from src.live.device import ADBDevice, LiveDevice, WindowsDesktopDevice
from src.live.runner import LiveMatchRunner, scaled_rect
from src.live.vision import mean_luma, mean_saturation
from src.viz import attach as viz
from src.viz import telemetry


def diagnose(config: LiveConfig, device: LiveDevice, out_path: Path) -> None:
    """Capture one screenshot and print the raw values match/ready detection uses.

    Point this at whatever screen is giving false positives (e.g. the home
    screen) and compare its numbers against a capture taken mid-match to pick
    a `match_indicator` rect and thresholds that actually separate the two.
    """
    image = device.screenshot()
    image.save(out_path)
    print(f"Captured {image.size[0]}x{image.size[1]} screenshot -> {out_path}")

    indicator_rect = scaled_rect(config, config.match_indicator, image.size)
    saturation = mean_saturation(image, indicator_rect)
    verdict = "IN MATCH" if saturation >= config.match_min_saturation else "not in match"
    print(
        f"match_indicator {indicator_rect}: saturation={saturation:.3f} "
        f"(threshold={config.match_min_saturation:.3f}) -> {verdict}"
    )

    for slot, region in enumerate(config.card_ready_regions):
        rect = scaled_rect(config, region, image.size)
        luma = mean_luma(image, rect)
        ready = "ready" if luma >= config.card_ready_min_luma else "not ready"
        print(f"card_ready_regions[{slot}] {rect}: luma={luma:.1f} (threshold={config.card_ready_min_luma:.1f}) -> {ready}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an already-running Clash Royale match through a calibrated display.")
    parser.add_argument("--config", type=Path, default=Path("configs/live_play.yaml"))
    parser.add_argument("--armed", action="store_true", help="Send clicks/taps to the configured target.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Capture one screenshot, print match/ready detection values for it, save it, and exit.",
    )
    parser.add_argument("--diagnose-out", type=Path, default=Path("live_diagnostic.png"))
    parser.add_argument(
        "--record", type=int, default=None, metavar="N",
        help="capture N frames to --record-out and exit, without tapping "
             "anything. Feed the result to `python -m src.live.replay` to "
             "exercise the whole perception stack offline, as many times as "
             "you like, without the game running.")
    parser.add_argument("--record-out", type=Path, default=Path("recordings/session"))
    parser.add_argument("--record-interval", type=float, default=0.05,
                        help="seconds between recorded frames (default 20fps). "
                             "Faster than `poll_seconds` on purpose: spell "
                             "detection needs frames close enough together to "
                             "watch a bloom grow.")
    parser.add_argument("--viz-port", type=int, default=None,
                        help="serve the 3D viewer on this port and mirror the "
                             "runner's output into its terminal pane "
                             "(http://localhost:PORT)")
    parser.add_argument("--viz-host", default="127.0.0.1",
                        help="bind address for --viz-port; loopback by default")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="trained `human`-tier policy to drive live play "
                             "(overrides `checkpoint:` in the config). Without "
                             "one the runner uses the configured heuristic.")
    args = parser.parse_args()
    config = load_live_config(args.config)
    device: LiveDevice
    if config.transport == "desktop":
        device = WindowsDesktopDevice(config.desktop_capture, config.window_title)
    else:
        device = ADBDevice(config.adb_path or "adb", config.device_serial)
    if args.diagnose:
        diagnose(config, device, args.diagnose_out)
        return
    if args.record:
        from src.live.replay import record

        record(device, args.record_out, frames=args.record,
               interval=args.record_interval)
        return
    log = print
    if args.viz_port:
        viz.start_server(args.viz_port, args.viz_host,
                         mode_note=f"streaming from live bridge "
                                   f"({'armed' if args.armed else 'observing'})",
                         activity="live")
        log = telemetry.TeeLogger()

    driver = _build_driver(config, args.checkpoint, log)
    if driver is not None:
        # No-op unless --viz-port opened a viewer. With one, the 3D graph
        # animates from the same forward passes that choose the real taps.
        viz.attach_live_driver(driver, label="live policy")

    runner = LiveMatchRunner(config, device, armed=args.armed, log=log, driver=driver)
    runner.run_forever()


def _build_driver(config, checkpoint_override, log):
    """Load the policy that will drive live play, or None for the heuristic."""
    checkpoint = checkpoint_override or config.checkpoint
    if not checkpoint:
        return None

    from src.agent.selfplay import (
        checkpoint_card_levels,
        checkpoint_deck,
        load_checkpoint,
    )
    from src.live.bridge import PolicyDriver
    from src.simulator.cards import load_arena, load_cards
    from src.simulator.levels import describe, scale_arena, scale_cards

    net, card_names = load_checkpoint(Path(checkpoint))
    # Card levels come from the checkpoint, not the live config: the policy
    # learned breakpoints at the levels it trained on, and running it against
    # a differently-levelled collection would silently be a different game.
    levels = checkpoint_card_levels(Path(checkpoint))
    log(f"Loaded {checkpoint} ({net.config.tier} tier); {describe(levels)}")
    cards = scale_cards(load_cards(), levels)
    arena = scale_arena(load_arena(), levels)

    deck = list(config.deck) or list(config.preset_deck)
    _warn_on_deck_mismatch(checkpoint_deck(Path(checkpoint)), deck, log)
    driver = PolicyDriver(net, card_names, cards, arena, deck)
    log(f"Policy driving live play with deck: {', '.join(deck)}")
    return driver


def _warn_on_deck_mismatch(trained, deck, log) -> None:
    """Say something when the policy has never played this deck.

    A warning and not a refusal, deliberately. Every checkpoint carries an
    embedding slot for all 171 cards and the deck is a runtime argument, so
    handing a policy an unfamiliar deck is legal and sometimes intended — a
    `full_pool` policy is *supposed* to play whatever it is given. What is
    not intended is finding out by watching it play badly, because a policy
    that has never seen these cards looks exactly like a policy that is
    simply bad.

    A checkpoint that records no deck was trained on a sampled pool, or
    predates this being recorded; the two are indistinguishable from here,
    so both get the same neutral note rather than a false alarm.
    """
    if trained is None:
        log("Checkpoint records no training deck (pooled stage, or saved "
            "before decks were recorded); cannot check it against yours.")
        return
    if set(trained) == set(deck):
        return
    unseen = sorted(set(deck) - set(trained))
    log(f"WARNING: this policy trained on [{', '.join(trained)}] but is "
        f"being given [{', '.join(deck)}]. "
        + (f"It has never played: {', '.join(unseen)}. " if unseen else "")
        + "It will run, and it will play worse than the numbers you "
          "measured. Train the `my_deck` stage on this deck, or accept it "
          "knowingly.")


if __name__ == "__main__":
    main()
