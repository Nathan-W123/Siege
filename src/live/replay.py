"""Run perception over recorded frames, with no game running and nothing armed.

Why this exists
---------------
Every other way to exercise the perception stack needs a live match, which
means the first time you find out whether it works is also the first time it
is tapping on your account. That is a bad place to discover that the
homography is off by a tile or that the plate never warms up.

So: record a couple of minutes of ordinary play once, then run the whole
stack over those frames as many times as you like, on any machine, changing
thresholds between runs. Nothing here opens a device, sends a tap, or needs
the game installed.

What it actually checks
-----------------------
The numbers to read first, in order, because each one makes the next
meaningless if it is wrong:

1. **Did the plate warm up?** If not, nothing downstream ran at all.
2. **Were units discovered?** Zero discoveries with a warm plate means the
   foreground threshold or the minimum area is wrong for this capture.
3. **How were teams decided?** `motion` is the trustworthy one. A run where
   most troops fall back to `bar` or `side` means tracks are being lost
   between frames — usually `max_drift_px` set too low for the capture rate.
4. **Were spells seen?** A capture polled slower than about four frames a
   second cannot resolve a bloom at all, and the report says so outright
   rather than letting a zero look like a quiet match.
5. **What got named, and why not?** The skip reasons are the interesting
   output. "ambiguous between [...]" is the system working correctly; "no
   deck prior yet" means it was not given one.

`--save-sprites` is the one to use when a number looks wrong but you cannot
tell why: it writes out every sprite that was actually cut, so you can look
at them. A harvest that is quietly slicing the top off every unit is obvious
in two seconds as a picture and nearly invisible as a statistic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.live.autolabel import SpriteHarvester
from src.live.background import PlateConfig, RunningPlate
from src.live.discover import DiscoverConfig, EntityDiscoverer
from src.live.spells import SpellConfig, SpellWatcher, resolve_identity

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def frame_paths(directory: Path | str) -> list[Path]:
    """Recorded frames, in capture order.

    Sorted by name, which is why `record` zero-pads: "frame_10.png" sorting
    before "frame_2.png" would feed the whole stack a shuffled match, and
    every motion-derived answer in it would be noise.
    """
    directory = Path(directory)
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in FRAME_SUFFIXES)


# ------------------------------------------------------------------ record


def record(device, out_dir: Path | str, frames: int = 400,
           interval: float = 0.05, log=print) -> Path:
    """Save `frames` captures to disk. Observes only — never taps.

    The default interval is 20fps rather than the bridge's `poll_seconds`
    (0.5s). Recording is the one place the capture rate is free, and spell
    detection needs frames close enough together to see a bloom grow, so
    there is no reason to record at the rate the decision loop happens to
    run at.
    """
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        device.screenshot().save(out_dir / f"frame_{i:06d}.png")
        if i and i % 100 == 0:
            log(f"  recorded {i}/{frames}")
        time.sleep(interval)
    log(f"Recorded {frames} frames -> {out_dir}")
    return out_dir


# ------------------------------------------------------------------ report


@dataclass
class ReplayReport:
    """What perception made of a recording."""

    frames: int = 0
    size: tuple[int, int] = (0, 0)
    plate_ready_at: int | None = None
    discoveries: int = 0
    by_kind: Counter = field(default_factory=Counter)
    by_team: Counter = field(default_factory=Counter)
    by_team_source: Counter = field(default_factory=Counter)
    spells: int = 0
    spell_radii: list[float] = field(default_factory=list)
    spell_names: Counter = field(default_factory=Counter)
    capture_interval: float = 0.0
    capture_too_slow: bool = False
    named: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "size": list(self.size),
            "plate_ready_at": self.plate_ready_at,
            "discoveries": self.discoveries,
            "by_kind": dict(self.by_kind),
            "by_team": dict(self.by_team),
            "by_team_source": dict(self.by_team_source),
            "spells": self.spells,
            "spell_names": dict(self.spell_names),
            "capture_interval": round(self.capture_interval, 3),
            "capture_too_slow": self.capture_too_slow,
            "named": dict(self.named),
            "skipped": dict(self.skipped),
        }


def format_report(report: ReplayReport) -> str:
    """The report as something worth reading in a terminal.

    Ordered so the first failure you hit is the first one that matters —
    a warm plate is a precondition for discoveries, and discoveries are a
    precondition for everything else.
    """
    lines = [
        f"frames            {report.frames} at {report.size[0]}x{report.size[1]}",
        f"capture interval  {report.capture_interval:.3f}s"
        + ("   TOO SLOW for spell detection" if report.capture_too_slow else ""),
    ]
    if report.plate_ready_at is None:
        lines.append("background plate  NEVER WARMED UP — nothing below ran")
        return "\n".join(lines)
    lines.append(f"background plate  ready at frame {report.plate_ready_at}")

    lines.append(f"discoveries       {report.discoveries}")
    if not report.discoveries:
        lines.append("   nothing found. With a warm plate this is a threshold "
                     "problem: try a lower min_delta or min_area.")
    else:
        lines.append(f"   kind   {dict(report.by_kind)}")
        lines.append(f"   team   {dict(report.by_team)}")
        lines.append(f"   decided by {dict(report.by_team_source)}")
        motion = report.by_team_source.get("motion", 0)
        if motion < report.discoveries / 2:
            lines.append("   most teams were NOT decided by motion — tracks are "
                         "being lost between frames (raise max_drift_px, or "
                         "record faster).")

    lines.append(f"spells            {report.spells} {dict(report.spell_names)}")
    if report.spell_radii:
        lines.append(f"   radii  {[round(r, 2) for r in report.spell_radii]} tiles")

    if report.named:
        lines.append(f"named             {dict(report.named)}")
    if report.skipped:
        lines.append("not named:")
        for reason, count in report.skipped.most_common():
            lines.append(f"   {count:4d}  {reason}")
    return "\n".join(lines)


# ------------------------------------------------------------------ replay


def replay(
    frames_dir: Path | str,
    cards: dict | None = None,
    homography=None,
    arena=None,
    deck: list[str] | None = None,
    interval: float = 0.05,
    plate_config: PlateConfig | None = None,
    discover_config: DiscoverConfig | None = None,
    spell_config: SpellConfig | None = None,
    save_sprites: Path | str | None = None,
) -> tuple[ReplayReport, SpriteHarvester | None]:
    """Run the whole perception stack over recorded frames.

    `deck` stands in for the opponent tracker, which cannot be reconstructed
    from pixels alone — it needs the play history the live runner feeds it.
    Supplying the eight cards models the *worst* case the tracker ever
    presents: the deck is known but the cycle is not, so nothing is ruled out
    by hand position or elixir. Whatever fraction gets named here is a floor,
    not an estimate.
    """
    paths = frame_paths(frames_dir)
    report = ReplayReport(frames=len(paths), capture_interval=interval)
    if not paths:
        return report, None

    plate = RunningPlate(plate_config or PlateConfig())
    discoverer = EntityDiscoverer(plate, homography=homography, arena=arena,
                                  config=discover_config or DiscoverConfig())
    watcher = SpellWatcher(homography, arena, config=spell_config or SpellConfig())
    harvester = SpriteHarvester(cards) if cards else None
    prior = _DeckPrior(deck) if deck else None

    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        if index == 0:
            report.size = image.size
        now = index * interval

        plate.update(image)
        if plate.ready and report.plate_ready_at is None:
            report.plate_ready_at = index

        for event in watcher.observe(image, now=now, homography=homography):
            report.spells += 1
            report.spell_radii.append(event.radius)
            name = resolve_identity(event, cards, candidates=deck) if cards else ""
            report.spell_names[name or "(unnamed)"] += 1

        found = discoverer.observe(image, now=now)
        for discovery in found:
            report.discoveries += 1
            report.by_kind[discovery.kind.value] += 1
            report.by_team[discovery.team] += 1
            report.by_team_source[discovery.team_source] += 1

        if harvester is not None and found:
            before = len(harvester.skipped)
            for sprite in harvester.observe(found, tracker=prior):
                report.named[sprite.card] += 1
            for reason in harvester.skipped[before:]:
                report.skipped[reason] += 1

    report.capture_too_slow = watcher.starved
    if save_sprites is not None and harvester is not None:
        _write_sprites(harvester, Path(save_sprites))
    return report, harvester


class _DeckPrior:
    """The tracker interface `SpriteHarvester` consumes, minus the cycle.

    A real `OpponentTracker` derives its hand and elixir from observed play,
    which a folder of frames does not contain. Rather than fake those, this
    reports the whole deck as the candidate set and no elixir constraint —
    deliberately the weakest prior the real tracker could ever present.
    """

    def __init__(self, deck: list[str]):
        self._deck = list(deck)
        self.elixir_range = (0.0, 10.0)

    def possible_hand(self) -> list[str]:
        return []

    def candidate_cards(self) -> list[str]:
        return list(self._deck)


def _write_sprites(harvester: SpriteHarvester, out_dir: Path) -> None:
    """Dump every harvested sprite as a transparent PNG, one folder per card.

    Transparent rather than on a background, because the alpha is the thing
    most likely to be quietly wrong and the thing you cannot check any other
    way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for card in harvester.library.cards:
        folder = out_dir / card
        folder.mkdir(exist_ok=True)
        for i, sprite in enumerate(harvester.library.get(card)):
            rgba = np.dstack([sprite.rgb,
                              (sprite.alpha * 255).astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(folder / f"{i:04d}_{sprite.facing}.png")


# --------------------------------------------------------------------- cli


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path, help="directory of recorded frames")
    parser.add_argument("--config", type=Path, default=None,
                        help="live_play.yaml, for the homography. Without it "
                             "positions stay in pixels and spells are dropped, "
                             "since a spell radius is meaningless untransformed.")
    parser.add_argument("--deck", nargs="*", default=None,
                        help="eight cards to use as the deck prior. Defaults to "
                             "`deck:` from --config.")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="seconds between recorded frames (default 20fps)")
    parser.add_argument("--save-sprites", type=Path, default=None,
                        help="write every harvested sprite here as a "
                             "transparent PNG, so you can look at them")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from src.simulator.cards import load_arena, load_cards

    cards, arena = load_cards(), load_arena()
    homography, deck = None, args.deck
    if args.config is not None:
        from src.live.config import load_live_config

        config = load_live_config(args.config)
        homography = config.homography()
        if homography is not None:
            first = frame_paths(args.frames)[:1]
            if first:
                with Image.open(first[0]) as probe:
                    homography = homography.scaled_to(config.reference_size, probe.size)
        if deck is None:
            deck = list(config.deck) or list(config.preset_deck)

    report, harvester = replay(
        args.frames, cards=cards, homography=homography, arena=arena,
        deck=deck, interval=args.interval, save_sprites=args.save_sprites)

    print(format_report(report))
    if harvester is not None and len(harvester.library):
        print(f"sprite library    {len(harvester.library)} sprites "
              f"across {len(harvester.library.cards)} cards")
    if homography is None:
        print("\nno --config, so no homography: spells were dropped and "
              "nothing was projected into arena tiles.")
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
