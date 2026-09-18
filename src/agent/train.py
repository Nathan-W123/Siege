"""PPO training CLI with curriculum stages and league self-play.

Examples:
  uv run python -m src.agent.train --run r1 --stage one_lane --steps 400000
  uv run python -m src.agent.train --run r1 --auto --bc-init checkpoints/bc_init.pt
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.agent import obs_layout
from src.agent.league import CheckpointPool
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.obs_noise import ObservationNoise, ObsNoiseConfig
from src.agent.ppo import (
    PPOConfig,
    RecurrentRolloutBuffer,
    RolloutBuffer,
    ppo_update,
    ppo_update_recurrent,
)
from src.agent import rewards as rewards_mod
from src.agent.exploiter import (
    fork_policy,
    frozen_main_bot,
    load_exploiter_config,
    retire_into_pool,
)
from src.agent.rewards import RewardShaper
from src.agent.selfplay import BotOpponent, PolicyBot, load_checkpoint, save_checkpoint
from src.bots.registry import get_bot
from src.decks.builder import AdaptiveDeckBuilder, AdaptiveDeckBuilderConfig
from src.decks.catalog import DeckCatalog
from src.training.focused_curriculum import FocusedRotationManager
from src.simulator.cards import load_arena, load_cards
from src.simulator.levels import describe, load_card_levels, scale_arena, scale_cards
from src.simulator.constants import MatchResult, Side
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv
from src.training.config import load_training_config
from src.training.curriculum import CurriculumStage, load_curriculum
from src.training.match_runner import run_match_detailed
from src.training.matchup_tracker import MatchupTracker
from src.viz import attach as viz
from src.viz import telemetry

DEVICE = torch.device("cpu")


def _report(line: str) -> None:
    """Print, and mirror to the 3D viewer's terminal when one is attached.

    `telemetry.log` is a no-op with no viewer, so this stays a plain print
    in the normal case.
    """
    print(line)
    telemetry.log(line)


class ShapedRewardFn:
    """Env-side reward: dense shaping + terminal outcome.

    One instance is shared across envs so the trainer can move `weight` once
    per rollout. Potential-based shaping needs per-episode state, so phi(s) is
    parked on the *engine* rather than on self — engines are per-env and are
    rebuilt on reset, which gives correct isolation for free.
    """

    def __init__(self, shaper: RewardShaper):
        self.shaper = shaper
        self.weight = shaper.config.anneal_start

    def __call__(self, events, side, engine) -> float:
        cfg = self.shaper.config
        terminated = engine.result != MatchResult.ONGOING
        if cfg.is_potential:
            # Both sides start with identical towers, so phi(s_0) = 0 exactly
            # — which is why a missing attribute can default to 0.0 rather
            # than needing a reset hook.
            attr = f"_potential_{int(side)}"
            prev = getattr(engine, attr, 0.0)
            curr = rewards_mod.tower_potential(engine, side, cfg)
            setattr(engine, attr, curr)
            # No `self.weight` here: potential shaping is policy-invariant, so
            # annealing it away would discard a free signal for no benefit.
            r = rewards_mod.potential_shaping(prev, curr, cfg, terminated)
        else:
            r = self.weight * self.shaper.compute_step_reward(events, side)
        if terminated:
            r += self.shaper.terminal_reward(engine.result, side)
        return r


def pinned_deck(stage: CurriculumStage, catalog: DeckCatalog) -> list[str] | None:
    """The agent's deck for this stage, when the stage pins one.

    None for a pooled stage, where the agent's deck is resampled every
    episode and so there is no single deck the checkpoint could claim to
    have been trained on. Recorded into the checkpoint as provenance, so the
    live bridge can warn when it is handed a different one.
    """
    return list(catalog.resolve(stage.deck)) if stage.deck else None


def resolve_stage_decks(stage: CurriculumStage, catalog: DeckCatalog, rng):
    if stage.deck:
        name_b = stage.deck
    else:
        pool_name = stage.agent_deck_pool or stage.deck_pool or "stage2_pool"
        name_b = str(rng.choice(catalog.pool(pool_name)))
    if stage.opponent_deck:
        name_t = stage.opponent_deck
    elif stage.opponent_deck_pool or stage.deck_pool:
        pool_name = stage.opponent_deck_pool or stage.deck_pool
        name_t = str(rng.choice(catalog.pool(pool_name)))
    else:
        name_t = name_b
    return name_b, name_t


class FocusedRotationState:
    """Ties FocusedRotationManager + AdaptiveDeckBuilder into the PPO loop.

    Both were originally written for the sequential match loop in
    `src/training/session.py` and were never reachable from PPO training, so
    the `focused_ladder` stage did not actually do focused rotation. They fit
    the PPO loop as-is because `CRBattleEnv.episode_metrics()` already
    reports everything they consume (`card_usage`, `win`, `crowns_for`,
    `elixir_spent`).

    Caveat worth knowing: with N parallel envs, episodes already in flight
    when the rotation advances still finish against the previous opponent
    deck, so the "one deck at a time" boundary is approximate rather than
    exact. That's a curriculum bias, not a correctness issue.
    """

    def __init__(self, stage, catalog, training_cfg):
        pool_name = (stage.opponent_deck_pool
                     or training_cfg.opponents.opponent_deck_pool)
        self.rotation = FocusedRotationManager(
            catalog.pool(pool_name), training_cfg.focused_rotation)
        adaptive = training_cfg.adaptive_deck
        self.builder = AdaptiveDeckBuilder(
            catalog,
            config=AdaptiveDeckBuilderConfig(
                rebuild_every_matches=adaptive.rebuild_every_matches if adaptive else 25,
                plateau_window=adaptive.plateau_window if adaptive else 20,
                plateau_threshold=adaptive.plateau_threshold if adaptive else 0.02,
            ),
        )

    def record(self, metrics: dict) -> None:
        won = bool(metrics["win"])
        self.rotation.record_result(won)
        self.builder.record_match(
            dict(metrics.get("card_usage", {})),
            won=won,
            crowns=int(metrics.get("crowns_for", 0)),
            elixir_spent=float(metrics.get("elixir_spent", 0.0)),
        )


def make_setup_fn(stage, catalog, training_cfg, pool, latest_bot,
                  matchup_tracker=None, focused=None):
    """Per-episode (deck_b, deck_t, opponent) sampler for one env.

    When `matchup_tracker` is given, scripted-bot selection is weakness-
    weighted (`opponents.weakness_weight` in configs/training.yaml) instead
    of uniform, so archetypes the agent is currently losing to more often —
    e.g. a persistent rusher/beatdown blind spot — get oversampled rather
    than diluted 1-in-N with bots it already beats.
    """
    scripted_names = list(stage.opponents) or list(training_cfg.opponents.scripted_bots)
    cfg = training_cfg.opponents
    # A bot's tracked `.name` (e.g. "champion_rusher") differs from the short
    # config key used to select it (e.g. "rusher") and doesn't depend on the
    # rng/deck passed in, so it's safe to resolve once up front and reuse it
    # to translate between the two consistently.
    _probe_rng = np.random.default_rng(0)
    display_name_of = {
        name: get_bot(name, catalog=catalog, rng=_probe_rng, skill_tier=cfg.skill_tier).name
        for name in scripted_names
    }

    def scripted(rng, deck_t_name):
        if matchup_tracker is not None and len(scripted_names) > 1:
            weights = matchup_tracker.sampling_weights(
                list(display_name_of.values()), cfg.weakness_weight)
            probs = np.array([weights[display_name_of[n]] for n in scripted_names], dtype=float)
            probs /= probs.sum()
            name = str(rng.choice(scripted_names, p=probs))
        else:
            name = str(rng.choice(scripted_names))
        # Archetype bots fight best on their archetype deck when it exists.
        deck_name = name if name in catalog.decks else deck_t_name
        bot = get_bot(name, catalog=catalog, deck_name=deck_name, rng=rng,
                      skill_tier=cfg.skill_tier)
        return BotOpponent(bot), deck_name

    def setup(rng):
        if focused is not None:
            # Focused rotation drives both decks: the agent plays the current
            # adaptive build, the opponent is whichever ladder deck the
            # rotation is currently working through.
            opp_name = focused.rotation.current_opponent_deck()
            bot = get_bot(str(rng.choice(scripted_names)), catalog=catalog,
                          deck_name=opp_name, rng=rng, skill_tier=cfg.skill_tier)
            return (focused.builder.current_deck(), catalog.resolve(opp_name),
                    BotOpponent(bot))
        name_b, name_t = resolve_stage_decks(stage, catalog, rng)
        if not stage.selfplay:
            opponent, name_t = scripted(rng, name_t)
        else:
            r = float(rng.random())
            if r < cfg.sample_scripted:
                opponent, name_t = scripted(rng, name_t)
            elif r < cfg.sample_scripted + cfg.sample_latest or not pool.members():
                opponent = latest_bot
            else:
                opponent = pool.sample_opponent(rng) or latest_bot
        return catalog.resolve(name_b), catalog.resolve(name_t), opponent

    return setup


def quick_eval(net, card_names, stage, catalog, arena, training_cfg,
               bots: list[str], matches: int, seed: int) -> dict[str, float]:
    """Deterministic policy vs each scripted bot; returns win rates
    (draws count half)."""
    policy = PolicyBot(net, card_names, name="eval", deterministic=True)
    rng = np.random.default_rng(seed)
    out = {}
    for bot_name in bots:
        deck_t_name = bot_name if bot_name in catalog.decks else \
            resolve_stage_decks(stage, catalog, rng)[1]
        score = 0.0
        for i in range(matches):
            name_b, _ = resolve_stage_decks(stage, catalog, rng)
            bot = get_bot(bot_name, catalog=catalog, deck_name=deck_t_name,
                          rng=rng, skill_tier=training_cfg.opponents.skill_tier)
            report = run_match_detailed(
                arena, catalog.resolve(name_b), catalog.resolve(deck_t_name),
                policy, bot, seed=int(rng.integers(2**31)),
                lanes=stage.single_lane or "both", regulation=stage.match_time)
            won = report.agent_won(Side.BOTTOM)
            score += 0.5 if won is None else float(won)
        out[bot_name] = score / matches
    return out


def train_exploiter(
    net,
    stage: CurriculumStage,
    *,
    card_names,
    catalog,
    arena,
    cards,
    training_cfg,
    ppo_cfg: PPOConfig,
    pool: CheckpointPool,
    exploiter_cfg,
    tier,
    obs_noise_cfg,
    n_envs: int,
    seed: int,
    global_step: int,
) -> Path:
    """Train a main-exploiter against a frozen snapshot and retire it.

    Deliberately *not* a curriculum stage: the exploiter faces one fixed
    opponent with no scripted mixing and no PFSP, because its whole job is to
    overfit to the current main agent and expose a specific hole. See
    `src/agent/exploiter.py` for why that is worth compute.
    """
    exploiter = fork_policy(net)
    frozen = frozen_main_bot(net, card_names)
    opponent = BotOpponent(frozen)
    optimizer = torch.optim.Adam(exploiter.parameters(), lr=ppo_cfg.lr)
    shaper = RewardShaper()
    reward_fn = ShapedRewardFn(shaper)
    reward_fn.weight = shaper.anneal_factor(global_step)

    def env_fn():
        deck = catalog.resolve(stage.deck or "training_mirror")
        return CRBattleEnv(cards, arena, deck, list(deck), reward_fn=reward_fn,
                           lanes=stage.single_lane or "both",
                           regulation=stage.match_time, opponent=opponent,
                           tier=tier,
                           with_units=getattr(net.config, "use_set_encoder", False),
                           critic_tier=getattr(net.config, "critic_tier", None))

    envs = SyncVecEnv([env_fn for _ in range(n_envs)])
    obs, masks = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
    mask_shapes = {k: v.shape[1:] for k, v in masks.items()}

    trained = 0
    while trained < exploiter_cfg.train_steps:
        buffer = RolloutBuffer(ppo_cfg.n_steps, n_envs, obs_shapes, mask_shapes)
        for _ in range(ppo_cfg.n_steps):
            actions, log_probs, values = exploiter.act(
                obs_to_tensors(obs, DEVICE), masks_to_tensors(masks, DEVICE))
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, next_masks, _ = envs.step(actions_np)
            buffer.add(obs, masks, actions_np, log_probs.cpu().numpy(),
                       values.cpu().numpy(), rewards, dones)
            obs, masks = next_obs, next_masks
        with torch.no_grad():
            _, _, last_values = exploiter.act(obs_to_tensors(obs, DEVICE),
                                              masks_to_tensors(masks, DEVICE))
        buffer.compute_returns(last_values.cpu().numpy(), ppo_cfg.gamma,
                               ppo_cfg.gae_lambda)
        # Hotter than the main agent on purpose: an exploiter is searching for
        # an unusual line that works, not playing the percentages.
        ppo_update(exploiter, optimizer, buffer, ppo_cfg,
                   ppo_cfg.ent_coef(global_step) * exploiter_cfg.ent_coef_scale,
                   DEVICE, rng)
        trained += ppo_cfg.n_steps * n_envs

    return retire_into_pool(exploiter, card_names, pool, global_step)


def train_stage(
    net,
    optimizer,
    stage: CurriculumStage,
    *,
    card_names,
    catalog,
    arena,
    cards,
    training_cfg,
    ppo_cfg: PPOConfig,
    pool: CheckpointPool,
    run_dir: Path,
    global_step: int,
    step_budget: int,
    n_envs: int,
    n_workers: int = 0,
    seed: int,
    tier=obs_layout.TIER_FULL,
    obs_noise_cfg: dict | None = None,
    matchup_tracker: MatchupTracker | None = None,
    card_levels=None,
) -> int:
    shaper = RewardShaper()
    reward_fn = ShapedRewardFn(shaper)
    # Recorded into every checkpoint this stage writes, so the live bridge
    # can tell whether the policy it is about to run has ever played the
    # deck it is being handed. None for a pooled stage.
    stage_deck = pinned_deck(stage, catalog)
    latest_bot = PolicyBot(net, card_names, name="latest", deterministic=False)
    focused = (FocusedRotationState(stage, catalog, training_cfg)
               if stage.focused_rotation else None)
    setup_fn = make_setup_fn(stage, catalog, training_cfg, pool, latest_bot,
                             matchup_tracker, focused)

    # Detection noise degrades *training* observations only (see #32 /
    # src.agent.obs_noise). `quick_eval` and the frozen benchmark build their
    # own clean engines, so eval numbers stay comparable across runs.
    noise_cfg = ObsNoiseConfig.from_dict(obs_noise_cfg)

    def make_env_fn(index: int):
        """One factory per env, with the noise seed derived from the index.

        A shared mutable counter would look equivalent but isn't: subprocess
        workers each receive their own cloudpickled copy of the closure, so
        every worker would restart the counter and envs in different workers
        would see *identical* detection noise. Deriving from the index keeps
        the streams distinct and reproducible however the envs are hosted.
        """

        def env_fn():
            deck = catalog.resolve(stage.deck or "training_mirror")
            noise = (ObservationNoise(noise_cfg, seed=seed + 9973 * (index + 1))
                     if noise_cfg.enabled and obs_layout.tier_uses_spatial(tier)
                     else None)
            return CRBattleEnv(cards, arena, deck, list(deck), reward_fn=reward_fn,
                               lanes=stage.single_lane or "both",
                               regulation=stage.match_time, setup_fn=setup_fn,
                               tier=tier, obs_noise=noise,
                               with_units=getattr(net.config, "use_set_encoder", False),
                               critic_tier=getattr(net.config, "critic_tier", None))

        return env_fn

    env_fns = [make_env_fn(i) for i in range(n_envs)]

    # Subprocess workers only pay off once each worker still holds enough envs
    # to keep the batched opponent forward wide: measured on an idle box,
    # 8 envs sees no benefit (and is worse at 2 workers), while 32 envs across
    # 8 workers gains ~34%. Default stays single-process.
    if n_workers:
        from src.simulator.subproc_vec_env import SubprocVecEnv
        envs = SubprocVecEnv(env_fns, n_workers=n_workers, live_bot=latest_bot)
    else:
        envs = SyncVecEnv(env_fns)
    obs, masks = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)

    league_cfg = training_cfg.raw.get("league", {})
    eval_cfg = training_cfg.raw.get("eval", {})
    snapshot_every = int(league_cfg.get("snapshot_every", 100_000))
    eval_every = int(eval_cfg.get("every_steps", 100_000))
    next_snapshot = global_step + snapshot_every
    next_eval = global_step + eval_every
    exploiter_cfg = load_exploiter_config(league_cfg.get("exploiter"))
    next_exploiter = global_step + exploiter_cfg.every_steps

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(run_dir / "tb"))
    except ImportError:
        pass
    for path, header in ((train_csv, ["step", "stage", "win_rate", "draw_rate",
                                      "reward_mean", "entropy", "policy_loss",
                                      "value_loss", "leak_per_match", "match_time",
                                      "card_usage_entropy", "steps_per_sec"]),
                         (eval_csv, ["step", "stage", "bot", "win_rate"])):
        if not path.exists():
            path.write_text(",".join(header) + "\n")

    stage_start = global_step
    obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
    mask_shapes = {k: v.shape[1:] for k, v in masks.items()}
    ep_metrics: list[dict] = []

    recurrent = net.config.use_recurrence
    # Memory persists across rollout boundaries — a match spans several
    # rollouts, so zeroing here would blind the policy every n_steps. It is
    # detached instead, which bounds the backward graph without truncating
    # the forward recurrence.
    hidden = net.initial_hidden(n_envs, DEVICE) if recurrent else None
    prev_dones = None

    # None unless a viewer is attached (`--viz-port`), in which case every
    # viz call below is a no-op.
    viz_probe = viz.attach_to_network(net, label=f"{stage.name} stage")

    while global_step - stage_start < step_budget:
        if recurrent:
            buffer = RecurrentRolloutBuffer(ppo_cfg.n_steps, n_envs, obs_shapes,
                                            mask_shapes, net.config.hidden_size)
            buffer.set_initial_hidden(hidden)
        else:
            buffer = RolloutBuffer(ppo_cfg.n_steps, n_envs, obs_shapes, mask_shapes)
        t0 = time.perf_counter()
        reward_sum = 0.0
        for rollout_step in range(ppo_cfg.n_steps):
            obs_t = obs_to_tensors(obs, DEVICE)
            masks_t = masks_to_tensors(masks, DEVICE)
            if recurrent:
                done_t = (torch.as_tensor(prev_dones, device=DEVICE)
                          if prev_dones is not None else None)
                actions, log_probs, values, hidden = net.act_recurrent(
                    obs_t, masks_t, hidden, done_t)
            else:
                actions, log_probs, values = net.act(obs_t, masks_t)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, next_masks, infos = envs.step(actions_np)
            prev_dones = dones
            buffer.add(obs, masks, actions_np, log_probs.cpu().numpy(),
                       values.cpu().numpy(), rewards, dones)
            reward_sum += float(rewards.sum())
            for info in infos:
                m = info.get("episode_metrics")
                if m:
                    ep_metrics.append(m)
                    opp = m["opponent"]
                    if opp.startswith(("step_", "anchor_")):
                        won = None if m["draw"] else bool(m["win"])
                        pool.record_result(f"{opp}.pt", won)
                    if matchup_tracker is not None and not m["draw"]:
                        matchup_tracker.record(opp, bool(m["win"]))
                    if focused is not None:
                        focused.record(m)
            obs, masks = next_obs, next_masks
            if rollout_step % viz.ACT_EVERY_STEPS == 0:
                viz.emit_act(viz_probe, global_step + rollout_step * n_envs)

        with torch.no_grad():
            if recurrent:
                done_t = (torch.as_tensor(prev_dones, device=DEVICE)
                          if prev_dones is not None else None)
                # Bootstrap value only — the returned hidden is discarded so
                # this peek does not advance the memory the next rollout
                # continues from.
                _, _, last_values, _ = net.act_recurrent(
                    obs_to_tensors(obs, DEVICE), masks_to_tensors(masks, DEVICE),
                    hidden, done_t)
            else:
                _, _, last_values = net.act(obs_to_tensors(obs, DEVICE),
                                            masks_to_tensors(masks, DEVICE))
        buffer.compute_returns(last_values.cpu().numpy(), ppo_cfg.gamma,
                               ppo_cfg.gae_lambda)
        if recurrent:
            stats = ppo_update_recurrent(net, optimizer, buffer, ppo_cfg,
                                         ppo_cfg.ent_coef(global_step), DEVICE, rng)
            # Truncated BPTT: carry the memory forward, drop the graph.
            hidden = hidden.detach()
        else:
            stats = ppo_update(net, optimizer, buffer, ppo_cfg,
                               ppo_cfg.ent_coef(global_step), DEVICE, rng)
        # Workers hold their own copy of the live policy, so self-play
        # opponents would otherwise keep playing the weights they were
        # spawned with. PPO does not mutate the net mid-rollout, so pushing
        # here reproduces single-process behaviour exactly rather than
        # approximating it.
        if hasattr(envs, "set_weights"):
            envs.set_weights(net.state_dict())
        global_step += ppo_cfg.n_steps * n_envs
        reward_fn.weight = shaper.anneal_factor(global_step)
        sps = ppo_cfg.n_steps * n_envs / (time.perf_counter() - t0)

        if ep_metrics:
            wins = np.mean([m["win"] for m in ep_metrics])
            draws = np.mean([m["draw"] for m in ep_metrics])
            leak = np.mean([m["elixir_leaked"] for m in ep_metrics])
            mtime = np.mean([m["match_time"] for m in ep_metrics])
            usage = {}
            for m in ep_metrics:
                for k, v in m["card_usage"].items():
                    usage[k] = usage.get(k, 0) + v
            total = sum(usage.values()) or 1
            probs = np.array([v / total for v in usage.values()])
            usage_entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        else:
            wins = draws = leak = mtime = usage_entropy = 0.0
        with train_csv.open("a", newline="") as f:
            csv.writer(f).writerow(
                [global_step, stage.name, f"{wins:.3f}", f"{draws:.3f}",
                 f"{reward_sum / (ppo_cfg.n_steps * n_envs):.4f}",
                 f"{stats['entropy']:.3f}", f"{stats['policy_loss']:.4f}",
                 f"{stats['value_loss']:.4f}", f"{leak:.2f}", f"{mtime:.0f}",
                 f"{usage_entropy:.3f}", f"{sps:.0f}"])
        if tb_writer is not None:
            tb_writer.add_scalar("train/win_rate", wins, global_step)
            tb_writer.add_scalar("train/reward_mean",
                                 reward_sum / (ppo_cfg.n_steps * n_envs), global_step)
            tb_writer.add_scalar("train/entropy", stats["entropy"], global_step)
            tb_writer.add_scalar("train/policy_loss", stats["policy_loss"], global_step)
            tb_writer.add_scalar("train/value_loss", stats["value_loss"], global_step)
            tb_writer.add_scalar("train/reward_anneal", reward_fn.weight, global_step)
            tb_writer.add_scalar("train/steps_per_sec", sps, global_step)
        _report(f"[{stage.name}] step {global_step:>8}  win {wins:.2f}  draw {draws:.2f}  "
                f"ent {stats['entropy']:.2f}  anneal {reward_fn.weight:.2f}  {sps:.0f} sps  "
                f"eps {len(ep_metrics)}")
        viz.emit_learn(viz_probe, global_step, stage=stage.name)
        viz.emit_stats("training", {
            "step": global_step, "win_rate": round(float(wins), 3),
            "entropy": round(stats["entropy"], 3),
            "policy_loss": round(stats["policy_loss"], 4),
            "value_loss": round(stats["value_loss"], 4),
            "sps": round(sps),
        })
        if focused is not None:
            p = focused.rotation.progress()
            _report(f"    focus: deck {p['deck_index'] + 1}/{p['deck_total']} "
                    f"{p['current_deck']}  vs-deck WR {p['current_deck_win_rate']:.2f} "
                    f"({p['matches_vs_current']} m)  overall {p['overall_win_rate']:.2f}  "
                    f"cycle {p['cycle']}")
            if tb_writer is not None:
                tb_writer.add_scalar("focus/deck_index", p["deck_index"], global_step)
                tb_writer.add_scalar("focus/overall_win_rate",
                                     p["overall_win_rate"], global_step)
                tb_writer.add_scalar("focus/cycle", p["cycle"], global_step)
        ep_metrics = []

        if global_step >= next_snapshot:
            pool.snapshot(net, card_names, global_step)
            pool.save_ledger()
            save_checkpoint(net, card_names, run_dir / "latest.pt",
                            card_levels=card_levels, deck=stage_deck)
            next_snapshot += snapshot_every

        if exploiter_cfg.active and global_step >= next_exploiter:
            print(f"[{stage.name}] training exploiter vs frozen main "
                  f"({exploiter_cfg.train_steps} steps)...")
            path = train_exploiter(
                net, stage,
                card_names=card_names, catalog=catalog, arena=arena, cards=cards,
                training_cfg=training_cfg, ppo_cfg=ppo_cfg, pool=pool,
                exploiter_cfg=exploiter_cfg, tier=tier, obs_noise_cfg=obs_noise_cfg,
                n_envs=n_envs, seed=seed + global_step, global_step=global_step)
            print(f"[{stage.name}] exploiter retired into pool: {path.name}")
            next_exploiter += exploiter_cfg.every_steps

        if global_step >= next_eval:
            bots = list(stage.opponents) or list(training_cfg.opponents.scripted_bots)
            scores = quick_eval(net, card_names, stage, catalog, arena, training_cfg,
                                bots, matches=20, seed=global_step)
            with eval_csv.open("a", newline="") as f:
                for bot, wr in scores.items():
                    csv.writer(f).writerow([global_step, stage.name, bot, f"{wr:.3f}"])
            if tb_writer is not None:
                for bot, wr in scores.items():
                    tb_writer.add_scalar(f"eval/win_rate/{bot}", wr, global_step)
            _report(f"[{stage.name}] eval @ {global_step}: "
                    + "  ".join(f"{b}={wr:.2f}" for b, wr in scores.items()))
            next_eval += eval_every

            promote = stage.promote
            if promote and scores.get(promote.vs, 0.0) >= promote.win_rate:
                confirm = quick_eval(net, card_names, stage, catalog, arena,
                                     training_cfg, [promote.vs],
                                     matches=promote.matches, seed=global_step + 1)
                if confirm[promote.vs] >= promote.win_rate:
                    _report(f"[{stage.name}] PROMOTED: {promote.vs} "
                            f"win rate {confirm[promote.vs]:.2f} "
                            f">= {promote.win_rate}")
                    save_checkpoint(net, card_names, run_dir / f"{stage.name}_final.pt",
                                    card_levels=card_levels, deck=stage_deck)
                    if tb_writer is not None:
                        tb_writer.close()
                    viz.detach(viz_probe)
                    return global_step
    viz.detach(viz_probe)
    save_checkpoint(net, card_names, run_dir / f"{stage.name}_final.pt",
                    card_levels=card_levels, deck=stage_deck)
    if tb_writer is not None:
        tb_writer.close()
    return global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run1")
    parser.add_argument("--stage", default=None, help="stage name; omit with --auto")
    parser.add_argument("--auto", action="store_true", help="advance through all stages")
    parser.add_argument("--steps", type=int, default=None,
                        help="per-stage step budget (default: ppo.total_steps)")
    parser.add_argument("--bc-init", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Intra-op torch threads (default 1; small nets "
                             "lose more to sync than they gain).")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Subprocess env workers (0 = single process). "
                             "Measured worthwhile only at high --n-envs: no "
                             "gain at 8 envs, ~34%% at 32 envs / 8 workers.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None,
                        help="training yaml path (default: configs/training.yaml)")
    parser.add_argument("--card-levels", default=None,
                        help="card-levels yaml (overrides the training config's "
                             "`card_levels:` block). Levels are baked into the "
                             "checkpoint so live play can reproduce them.")
    parser.add_argument("--global-step-start", type=int, default=0,
                        help="True cumulative step count so far, for a --resume that "
                             "continues an existing run under a new --run name. Without "
                             "this, reward-anneal and entropy schedules (both keyed off "
                             "global_step) incorrectly restart from their step-0 values "
                             "even though the resumed network is already deep into "
                             "training — e.g. dense reward shaping would switch back on "
                             "for a policy that should be past it.")
    parser.add_argument("--viz-port", type=int, default=None,
                        help="serve the 3D network viewer on this port for the "
                             "duration of the run (http://localhost:PORT). "
                             "Off by default; when off the telemetry calls in "
                             "the training loop are no-ops.")
    parser.add_argument("--viz-host", default="127.0.0.1",
                        help="bind address for --viz-port; loopback by default")
    args = parser.parse_args()

    viz.start_server(args.viz_port, args.viz_host,
                     mode_note=f"streaming from training run {args.run!r}",
                     activity="training")

    # These nets are small and the batches are narrow, so intra-op threading
    # costs more in synchronization than it recovers in arithmetic. Measured
    # per policy step at 8 envs: 7.0 ms with 1 thread, 10.1 ms with 16, and a
    # pathological 67.8 ms with 4. Overridable, but 1 is the right default.
    torch.set_num_threads(max(1, args.torch_threads))

    torch.manual_seed(args.seed)
    training_cfg = load_training_config(Path(args.config) if args.config else None)
    # Card levels are chosen *before* anything loads a stat, because every
    # deck, bot and env downstream reads these objects. Scaling later would
    # leave some component playing a different game.
    card_levels = (load_card_levels(path=args.card_levels) if args.card_levels
                   else load_card_levels(training_cfg.raw.get("card_levels")))
    print(f"[train] {describe(card_levels)}")
    cards = scale_cards(load_cards(), card_levels)
    arena = scale_arena(load_arena(), card_levels)
    catalog = DeckCatalog(cards=cards)
    raw_ppo = training_cfg.raw.get("ppo", {})
    ppo_cfg = PPOConfig(
        n_steps=int(raw_ppo.get("n_steps", 512)),
        batch_size=int(raw_ppo.get("batch_size", 1024)),
        n_epochs=int(raw_ppo.get("n_epochs", 4)),
        lr=float(raw_ppo.get("lr", 3e-4)),
        gamma=float(raw_ppo.get("gamma", 0.997)),
        gae_lambda=float(raw_ppo.get("gae_lambda", 0.95)),
        clip_range=float(raw_ppo.get("clip_range", 0.2)),
        vf_coef=float(raw_ppo.get("vf_coef", 0.5)),
        ent_coef_start=float(raw_ppo.get("ent_coef_start", 0.01)),
        ent_coef_end=float(raw_ppo.get("ent_coef_end", 0.002)),
        max_grad_norm=float(raw_ppo.get("max_grad_norm", 0.5)),
        total_steps=int(raw_ppo.get("total_steps", 3_000_000)),
    )
    n_envs = args.n_envs or int(raw_ppo.get("n_envs", 8))
    # 0 = single-process. Only worth enabling at high env counts; see the
    # note in train_stage and scripts/bench_throughput.py.
    n_workers = (args.n_workers if args.n_workers is not None
                 else int(raw_ppo.get("n_workers", 0)))
    card_names = list(cards.keys())

    if args.resume:
        net, card_names = load_checkpoint(Path(args.resume))
        net.train()
    elif args.bc_init:
        net, card_names = load_checkpoint(Path(args.bc_init))
        net.train()
        print(f"[train] initialized from BC checkpoint {args.bc_init}")
    else:
        net = make_network(len(card_names),
                           training_cfg.raw.get("network"))
    tier = net.config.tier
    obs_noise_cfg = training_cfg.raw.get("obs_noise")
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr)

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    league_cfg = training_cfg.raw.get("league", {})
    pool = CheckpointPool(Path("checkpoints") / args.run,
                          pool_max=int(league_cfg.get("pool_max", 20)),
                          anchor_every=int(league_cfg.get("anchor_every", 500_000)))

    stages = load_curriculum()
    if args.auto:
        selected = stages
    else:
        name = args.stage or stages[0].name
        selected = [s for s in stages if s.name == name]
        if not selected:
            raise SystemExit(f"unknown stage {name}")

    global_step = args.global_step_start
    budget = args.steps or ppo_cfg.total_steps
    for stage in selected:
        print(f"=== stage {stage.name} (budget {budget}) ===")
        global_step = train_stage(
            net, optimizer, stage,
            card_names=card_names, catalog=catalog, arena=arena, cards=cards,
            training_cfg=training_cfg, ppo_cfg=ppo_cfg, pool=pool,
            run_dir=run_dir, global_step=global_step, step_budget=budget,
            n_envs=n_envs, n_workers=n_workers, seed=args.seed, tier=tier,
            obs_noise_cfg=obs_noise_cfg)
    save_checkpoint(net, card_names, run_dir / "final.pt", card_levels=card_levels,
                    deck=pinned_deck(selected[-1], catalog) if selected else None)
    print(f"[train] done at step {global_step}; saved {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
