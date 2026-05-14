import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ppo_assignment.agent import AutoregressiveAssignmentPPOAgent
    from ppo_assignment.device import resolve_device
    from ppo_assignment.rollout import MacroTransition
else:
    from .agent import AutoregressiveAssignmentPPOAgent
    from .device import resolve_device
    from .rollout import MacroTransition

try:
    from foraging.foraging import ForagingEnvironment
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from foraging import ForagingEnvironment


_WORKER_ENV_ARGS = None
_WORKER_AGENT = None


ROLLOUT_METRIC_COUNT_KEYS = [
    "goal_choice_count",
    "decision_trigger_object_collected",
    "decision_trigger_max_plan_steps",
    "remaining_objects_at_episode_end_sum",
    "episode_end_count",
]


ROLLOUT_METRIC_FIELDNAMES = [
    "decision_trigger_object_collected",
    "decision_trigger_max_plan_steps",
    "remaining_objects_at_episode_end",
]


class RandomAgent:
    def action(self, state, deterministic=True):
        _, agent_locations = state
        return [random.randrange(0, 4) for _ in agent_locations]

    def reward(self, reward):
        pass

    def reset_episode_state(self):
        pass

    def finish_episode(self, final_state):
        pass


def run_episode(env, agent, deterministic=False, train=True):
    state = env.reset()
    if hasattr(agent, "reset_episode_state"):
        agent.reset_episode_state()

    total_reward = 0.0
    steps = 0
    start_macros = len(agent.rollout_buffer) if hasattr(agent, "rollout_buffer") else 0

    while not env.done():
        actions = agent.action(state, deterministic=deterministic)
        reward, world, agent_locations = env.perform_actions(actions)
        next_state = (world, agent_locations)
        agent.reward(reward)
        total_reward += reward
        steps += 1
        state = next_state

    if hasattr(agent, "finish_episode"):
        agent.finish_episode(state)

    end_macros = len(agent.rollout_buffer) if hasattr(agent, "rollout_buffer") else start_macros
    macro_count = max(end_macros - start_macros, 0)
    return total_reward, steps, macro_count


def validate_env_args(h, w, objects, agents):
    if h <= 0 or w <= 0:
        raise ValueError(f"Environment dimensions must be positive, got h={h}, w={w}.")
    if agents < 2:
        raise ValueError(f"ForagingEnvironment requires at least 2 agents, got {agents}.")
    if objects < 1:
        raise ValueError(f"At least 1 object is required, got {objects}.")
    cells = h * w
    if agents + objects > cells:
        raise ValueError(
            f"Too many entities for board: agents + objects = {agents + objects}, cells = {cells}."
        )


def normalize_sampling_args(args):
    defaults = {
        "min_height": args.height,
        "max_height_train": args.height,
        "min_width": args.width,
        "max_width_train": args.width,
        "min_agents": args.agents,
        "max_agents_train": args.agents,
        "min_objects": args.objects,
        "max_objects_train": args.objects,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def make_sampler_config(args):
    return {
        "min_height": args.min_height,
        "max_height_train": args.max_height_train,
        "min_width": args.min_width,
        "max_width_train": args.max_width_train,
        "min_agents": args.min_agents,
        "max_agents_train": args.max_agents_train,
        "min_objects": args.min_objects,
        "max_objects_train": args.max_objects_train,
        "min_occupancy": args.min_occupancy,
        "max_occupancy": args.max_occupancy,
        "min_objects_per_agent": args.min_objects_per_agent,
        "max_objects_per_agent": args.max_objects_per_agent,
    }


def sample_env_args(config):
    for _ in range(1000):
        h = random.randint(config["min_height"], config["max_height_train"])
        w = random.randint(config["min_width"], config["max_width_train"])
        cells = h * w

        min_total = math.ceil(config["min_occupancy"] * cells)
        max_total = math.floor(config["max_occupancy"] * cells)

        max_agents = min(config["max_agents_train"], max_total - 1)
        if max_agents < config["min_agents"]:
            continue

        agents = random.randint(config["min_agents"], max_agents)

        min_objects = max(
            config["min_objects"],
            math.ceil(config["min_objects_per_agent"] * agents),
            min_total - agents,
        )
        max_objects = min(
            config["max_objects_train"],
            math.floor(config["max_objects_per_agent"] * agents),
            cells - agents,
            max_total - agents,
        )

        if max_objects < min_objects:
            continue

        objects = random.randint(min_objects, max_objects)
        validate_env_args(h, w, objects, agents)
        return h, w, objects, agents

    raise RuntimeError("Could not sample a valid environment after 1000 attempts.")


def make_env_args(env_source):
    if isinstance(env_source, dict):
        return sample_env_args(env_source)
    return env_source


def empty_rollout_metric_counts():
    return {key: 0 for key in ROLLOUT_METRIC_COUNT_KEYS}


def add_rollout_metric_counts(total, chunk):
    for key in ROLLOUT_METRIC_COUNT_KEYS:
        total[key] = total.get(key, 0) + chunk.get(key, 0)
    return total


def finalize_rollout_metrics(counts):
    episode_count = counts.get("episode_end_count", 0)
    return {
        "decision_trigger_object_collected": counts.get("decision_trigger_object_collected", 0),
        "decision_trigger_max_plan_steps": counts.get("decision_trigger_max_plan_steps", 0),
        "remaining_objects_at_episode_end": 0.0
        if episode_count == 0
        else counts.get("remaining_objects_at_episode_end_sum", 0) / episode_count,
    }


def evaluate_agent(env_args, agent, episodes, deterministic=True, seed=None):
    random_state = None
    torch_state = None
    cuda_states = None
    if seed is not None:
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        random.seed(seed)
        torch.manual_seed(seed)

    saved_transitions = None
    if hasattr(agent, "rollout_buffer"):
        saved_transitions = list(agent.rollout_buffer.transitions)
        agent.rollout_buffer.clear()

    rewards = []
    lengths = []
    macros = []
    try:
        for _ in range(episodes):
            env = ForagingEnvironment(*make_env_args(env_args))
            reward, length, macro_count = run_episode(
                env,
                agent,
                deterministic=deterministic,
                train=False,
            )
            rewards.append(reward)
            lengths.append(length)
            macros.append(macro_count)
            if hasattr(agent, "rollout_buffer"):
                agent.rollout_buffer.clear()
    finally:
        if saved_transitions is not None:
            agent.rollout_buffer.transitions = saved_transitions
        if seed is not None:
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return {
        "avg_reward": sum(rewards) / len(rewards),
        "avg_length": sum(lengths) / len(lengths),
        "avg_macros": sum(macros) / len(macros),
    }


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def make_agent_kwargs(args, device):
    return {
        "w": args.max_width_train,
        "h": args.max_height_train,
        "agents": args.max_agents_train,
        "device": device,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "lr": args.lr,
        "clip_eps": args.clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "ppo_epochs": args.ppo_epochs,
        "max_plan_steps": args.max_plan_steps,
    }


def init_rollout_worker(env_args, agent_kwargs, torch_threads):
    global _WORKER_ENV_ARGS
    global _WORKER_AGENT

    if torch_threads is not None and torch_threads > 0:
        torch.set_num_threads(torch_threads)

    _WORKER_ENV_ARGS = env_args
    _WORKER_AGENT = AutoregressiveAssignmentPPOAgent(**agent_kwargs)


def tensor_to_plain(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    return x


def transition_to_plain(t):
    return {
        "decision_state": copy.deepcopy(t.decision_state),
        "agent_order": list(t.agent_order),
        "targets": copy.deepcopy(t.targets),
        "target_indices": copy.deepcopy(t.target_indices),
        "old_joint_logprob": tensor_to_plain(t.old_joint_logprob),
        "old_value": tensor_to_plain(t.old_value),
        "old_entropy": tensor_to_plain(t.old_entropy),
        "macro_reward": float(t.macro_reward),
        "duration": int(t.duration),
        "next_decision_state": copy.deepcopy(t.next_decision_state),
        "done": bool(t.done),
        "mode": t.mode,
        "visible_agent_indices": copy.deepcopy(t.visible_agent_indices),
        "filtered_goals": copy.deepcopy(t.filtered_goals),
        "next_visible_agent_indices": copy.deepcopy(t.next_visible_agent_indices),
        "next_filtered_goals": copy.deepcopy(t.next_filtered_goals),
    }


def transition_from_plain(d):
    return MacroTransition(
        decision_state=d["decision_state"],
        agent_order=d["agent_order"],
        targets=d["targets"],
        target_indices=d["target_indices"],
        old_joint_logprob=float(d["old_joint_logprob"]),
        old_value=float(d["old_value"]),
        old_entropy=float(d["old_entropy"]),
        macro_reward=float(d["macro_reward"]),
        duration=int(d["duration"]),
        next_decision_state=d["next_decision_state"],
        done=bool(d["done"]),
        mode=d.get("mode", "normal"),
        visible_agent_indices=d.get("visible_agent_indices"),
        filtered_goals=d.get("filtered_goals"),
        next_visible_agent_indices=d.get("next_visible_agent_indices"),
        next_filtered_goals=d.get("next_filtered_goals"),
    )


def rollout_worker(actor_state, critic_state, seed, episodes, worker_transport):
    random.seed(seed)
    torch.manual_seed(seed)

    if _WORKER_AGENT is None or _WORKER_ENV_ARGS is None:
        raise RuntimeError("Rollout worker was not initialized.")

    _WORKER_AGENT.actor.load_state_dict(actor_state)
    _WORKER_AGENT.critic.load_state_dict(critic_state)
    _WORKER_AGENT.rollout_buffer.clear()
    _WORKER_AGENT.reset_rollout_metrics()

    rewards = []
    lengths = []
    macros = []
    for _ in range(episodes):
        env = ForagingEnvironment(*make_env_args(_WORKER_ENV_ARGS))
        reward, length, macro_count = run_episode(env, _WORKER_AGENT, deterministic=False, train=True)
        rewards.append(reward)
        lengths.append(length)
        macros.append(macro_count)

    transitions = list(_WORKER_AGENT.rollout_buffer.transitions)
    if worker_transport == "plain":
        transitions = [transition_to_plain(transition) for transition in transitions]
    return rewards, lengths, macros, transitions, _WORKER_AGENT.get_rollout_metrics()


def episode_chunks(total_episodes, workers, requested_chunk_size):
    if requested_chunk_size > 0:
        chunk_size = requested_chunk_size
    else:
        chunk_size = 1
    chunks = []
    remaining = total_episodes
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def collect_rollouts_serial(env_args, agent, args, progress, update, last_eval_reward):
    agent.reset_rollout_metrics()
    rewards = []
    lengths = []
    macros = []
    for episode in range(1, args.episodes_per_update + 1):
        env = ForagingEnvironment(*make_env_args(env_args))
        reward, length, macro_count = run_episode(env, agent, deterministic=False, train=True)
        rewards.append(reward)
        lengths.append(length)
        macros.append(macro_count)
        progress.update(1)
        progress.set_postfix(
            update=f"{update}/{getattr(args, 'end_update', args.updates)}",
            ep=f"{episode}/{args.episodes_per_update}",
            reward=f"{reward:.2f}",
            length=length,
            macros=macro_count,
            workers=1,
            eval="-" if last_eval_reward is None else f"{last_eval_reward:.2f}",
        )
    return rewards, lengths, macros, agent.get_rollout_metrics()


def collect_rollouts_parallel(agent, args, progress, update, last_eval_reward, executor):
    actor_state = cpu_state_dict(agent.actor)
    critic_state = cpu_state_dict(agent.critic)
    chunks = episode_chunks(args.episodes_per_update, args.workers, args.worker_chunk_size)

    rewards = []
    lengths = []
    macros = []
    rollout_metric_counts = empty_rollout_metric_counts()
    completed = 0
    next_chunk_idx = 0
    pending = set()

    def submit_next_chunk():
        nonlocal next_chunk_idx
        if next_chunk_idx >= len(chunks):
            return
        chunk_idx = next_chunk_idx
        next_chunk_idx += 1
        pending.add(
            executor.submit(
                rollout_worker,
                actor_state,
                critic_state,
                args.seed + update * 100_000 + chunk_idx * 1_003,
                chunks[chunk_idx],
                args.worker_transport,
            )
        )

    for _ in range(min(args.workers, len(chunks))):
        submit_next_chunk()

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            chunk_rewards, chunk_lengths, chunk_macros, chunk_transitions, chunk_metrics = future.result()
            if args.worker_transport == "plain":
                chunk_transitions = [transition_from_plain(transition) for transition in chunk_transitions]
            agent.rollout_buffer.transitions.extend(chunk_transitions)
            add_rollout_metric_counts(rollout_metric_counts, chunk_metrics)
            rewards.extend(chunk_rewards)
            lengths.extend(chunk_lengths)
            macros.extend(chunk_macros)
            completed += len(chunk_rewards)
            progress.update(len(chunk_rewards))
            progress.set_postfix(
                update=f"{update}/{getattr(args, 'end_update', args.updates)}",
                ep=f"{completed}/{args.episodes_per_update}",
                reward=f"{chunk_rewards[-1]:.2f}",
                length=chunk_lengths[-1],
                macros=chunk_macros[-1],
                workers=args.workers,
                scheduler="dynamic" if args.worker_chunk_size == 0 else "chunks",
                eval="-" if last_eval_reward is None else f"{last_eval_reward:.2f}",
            )
            submit_next_chunk()

    return rewards, lengths, macros, rollout_metric_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Train event-driven autoregressive PPO for foraging.")
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--min-height", type=int, default=None)
    parser.add_argument("--max-height-train", type=int, default=None)
    parser.add_argument("--min-width", type=int, default=None)
    parser.add_argument("--max-width-train", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--max-agents-train", type=int, default=None)
    parser.add_argument("--min-objects", type=int, default=None)
    parser.add_argument("--max-objects-train", type=int, default=None)
    parser.add_argument("--min-occupancy", type=float, default=0.25)
    parser.add_argument("--max-occupancy", type=float, default=0.65)
    parser.add_argument("--min-objects-per-agent", type=float, default=1.5)
    parser.add_argument("--max-objects-per-agent", type=float, default=3.0)
    parser.add_argument("--updates", type=int, default=1000, help="Updates to run. With --resume/--resume-from, this is the number of additional updates.")
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed used for evaluation episodes. Set this for a fixed held-out eval sequence across updates.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--max-plan-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None, help="Name for runs/<run-name> outputs.")
    parser.add_argument("--runs-dir", default="runs", help="Base directory for run outputs.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to runs/<run-name>/checkpoints/latest.pt.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the checkpoint path before training and continue update numbering/logging when possible.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint to load before training. Saves future latest checkpoints to --checkpoint, or runs/<run-name>/checkpoints/latest.pt by default.",
    )
    parser.add_argument("--keep-best-checkpoints", type=int, default=5, help="Keep the top N eval checkpoints by reward. Use 0 to disable.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress reporting.")
    parser.add_argument("--print-every", type=int, default=0, help="Also print update stats every N updates.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU rollout workers. Use 1 for serial rollouts.")
    parser.add_argument("--worker-chunk-size", type=int, default=0, help="Episodes per worker task. 0 uses dynamic one-episode tasks.")
    parser.add_argument("--worker-torch-threads", type=int, default=1, help="Torch intra-op threads per rollout worker.")
    parser.add_argument(
        "--worker-transport",
        choices=["torch", "plain"],
        default="plain",
        help="How rollout workers return transitions. 'plain' converts tensors to Python floats/lists to avoid torch multiprocessing shared-memory mmap issues.",
    )
    parser.add_argument("--strict-device", action="store_true", help="Fail instead of falling back when the requested device is unavailable.")
    return parser.parse_args()


def create_run_paths(args):
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_assignment_{timestamp}"

    run_dir = Path(args.runs_dir) / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "latest.pt"
    os.makedirs(checkpoint_path.parent, exist_ok=True)

    return run_name, run_dir, log_dir, checkpoint_path


def write_config(log_dir, args, run_name, checkpoint_path, resume_path=None):
    config = vars(args).copy()
    config["run_name"] = run_name
    config["checkpoint_path"] = str(checkpoint_path)
    config["resume_path"] = "" if resume_path is None else str(resume_path)
    with open(log_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def read_last_metrics_update(metrics_path):
    if not metrics_path.exists():
        return 0, None

    last_update = 0
    last_eval_reward = None
    with open(metrics_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                last_update = max(last_update, int(row.get("update") or 0))
            except ValueError:
                pass
            eval_reward = row.get("eval_avg_reward")
            if eval_reward not in (None, ""):
                try:
                    last_eval_reward = float(eval_reward)
                except ValueError:
                    pass
    return last_update, last_eval_reward


def update_from_checkpoint_name(path):
    match = re.search(r"(?:^|_)update_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def make_metrics_writer(log_dir, append=False):
    metrics_path = log_dir / "metrics.csv"
    fieldnames = [
        "update",
        "episodes",
        "workers",
        "worker_chunk_size",
        "device",
        "avg_reward",
        "avg_length",
        "avg_macros",
        "avg_macro_duration",
        "loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "eval_avg_reward",
        "eval_avg_length",
        "eval_avg_macros",
        "rollout_sec",
        "ppo_sec",
        "eval_sec",
        "checkpoint_sec",
        "total_update_sec",
        "episodes_per_sec",
        "rollout_pct",
        "ppo_pct",
        "eval_pct",
        "checkpoint_pct",
        "recompute_logprob_sec",
        "critic_value_sec",
        "num_macro_transitions",
        "avg_candidate_count",
        "min_candidate_count",
        "max_candidate_count",
        "recompute_batches",
        "avg_recompute_batch_size",
        *ROLLOUT_METRIC_FIELDNAMES,
    ]
    mode = "w"
    write_header = True
    if append and metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header == fieldnames:
            mode = "a"
            write_header = False
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = log_dir / f"metrics_resume_{timestamp}.csv"

    handle = open(metrics_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    handle.flush()
    return handle, writer, metrics_path


def pct(part, total):
    return 0.0 if total <= 0 else 100.0 * part / total


def save_best_checkpoint(agent, checkpoint_dir, update, eval_reward, best_checkpoints, keep_best):
    if keep_best <= 0:
        return best_checkpoints

    checkpoint_path = checkpoint_dir / f"best_update_{update:04d}_reward_{eval_reward:.3f}.pt"
    agent.save(checkpoint_path, update=update, eval_reward=eval_reward)
    best_checkpoints.append((eval_reward, checkpoint_path))
    best_checkpoints.sort(key=lambda item: item[0], reverse=True)

    for _, path in best_checkpoints[keep_best:]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return best_checkpoints[:keep_best]


def main():
    args = parse_args()
    normalize_sampling_args(args)
    validate_env_args(args.height, args.width, args.objects, args.agents)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = str(resolve_device(args.device, strict=args.strict_device))

    run_name, run_dir, log_dir, checkpoint_path = create_run_paths(args)
    resume_path = Path(args.resume_from) if args.resume_from else (checkpoint_path if args.resume else None)
    write_config(log_dir, args, run_name, checkpoint_path, resume_path=resume_path)
    previous_metrics_update, previous_eval_reward = read_last_metrics_update(log_dir / "metrics.csv")
    metrics_handle, metrics_writer, metrics_path = make_metrics_writer(log_dir, append=resume_path is not None)
    print(f"run_name={run_name}")
    print(f"run_dir={run_dir}")
    print(f"metrics_log={metrics_path}")
    print(f"checkpoint={checkpoint_path}")
    if resume_path is not None:
        print(f"resume_from={resume_path}")
    print(f"device={args.device}")

    env_sampler_config = make_sampler_config(args)
    env_args = env_sampler_config
    agent = AutoregressiveAssignmentPPOAgent(
        w=args.max_width_train,
        h=args.max_height_train,
        agents=args.max_agents_train,
        device=args.device,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        max_plan_steps=args.max_plan_steps,
    )

    loaded_checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        loaded_checkpoint = agent.load(resume_path, load_optimizer=True)
        print(f"loaded_checkpoint={resume_path}")

    baseline = evaluate_agent(env_args, RandomAgent(), args.eval_episodes, seed=args.eval_seed)
    print(
        "random_baseline "
        f"avg_reward={baseline['avg_reward']:.3f} "
        f"avg_length={baseline['avg_length']:.1f}"
    )

    checkpoint_update = int(loaded_checkpoint.get("update", 0)) if loaded_checkpoint else 0
    checkpoint_name_update = update_from_checkpoint_name(resume_path) if resume_path is not None else 0
    if checkpoint_update > 0:
        completed_updates = checkpoint_update
    elif checkpoint_name_update > 0:
        completed_updates = checkpoint_name_update
    else:
        completed_updates = previous_metrics_update if resume_path is not None else 0
    start_update = completed_updates + 1
    end_update = completed_updates + args.updates
    args.end_update = end_update
    last_eval_reward = previous_eval_reward
    best_checkpoints = []
    checkpoint_dir = checkpoint_path.parent
    total_episodes = args.updates * args.episodes_per_update
    progress = tqdm(
        total=total_episodes,
        desc="training PPO",
        unit="episode",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    rollout_executor = None
    if args.workers > 1:
        rollout_executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_rollout_worker,
            initargs=(
                env_args,
                make_agent_kwargs(args, device="cpu"),
                args.worker_torch_threads,
            ),
        )

    try:
        for update in range(start_update, end_update + 1):
            update_start = time.perf_counter()
            agent.rollout_buffer.clear()
            rollout_start = time.perf_counter()
            if args.workers > 1:
                rewards, lengths, macros, rollout_metric_counts = collect_rollouts_parallel(
                    agent,
                    args,
                    progress,
                    update,
                    last_eval_reward,
                    rollout_executor,
                )
            else:
                rewards, lengths, macros, rollout_metric_counts = collect_rollouts_serial(
                    env_args,
                    agent,
                    args,
                    progress,
                    update,
                    last_eval_reward,
                )
            rollout_sec = time.perf_counter() - rollout_start

            progress.set_description(f"training PPO update {update}/{end_update}")
            ppo_start = time.perf_counter()
            stats = agent.train_update()
            stats.update(finalize_rollout_metrics(rollout_metric_counts))
            ppo_sec = time.perf_counter() - ppo_start
            avg_reward = sum(rewards) / len(rewards)
            avg_length = sum(lengths) / len(lengths)
            avg_macros = sum(macros) / len(macros)
            avg_duration = stats.get("avg_macro_duration", 0.0)
            eval_sec = 0.0
            checkpoint_sec = 0.0
            eval_stats = None
            progress.set_postfix(
                update=f"{update}/{end_update}",
                reward=f"{avg_reward:.2f}",
                length=f"{avg_length:.1f}",
                macros=f"{avg_macros:.1f}",
                macro_len=f"{avg_duration:.2f}",
                loss=f"{stats.get('loss', 0.0):.3f}",
                eval="-" if last_eval_reward is None else f"{last_eval_reward:.2f}",
            )

            if args.print_every and update % args.print_every == 0:
                tqdm.write(
                    f"update={update:04d} "
                    f"avg_reward={avg_reward:.3f} "
                    f"avg_length={avg_length:.1f} "
                    f"avg_macros={avg_macros:.1f} "
                    f"avg_macro_duration={avg_duration:.2f} "
                    f"loss={stats.get('loss', 0.0):.4f}"
                )

            if update % args.eval_every == 0 or update == end_update:
                progress.set_description(f"evaluating update {update}/{end_update}")
                eval_start = time.perf_counter()
                eval_stats = evaluate_agent(env_args, agent, args.eval_episodes, seed=args.eval_seed)
                eval_sec = time.perf_counter() - eval_start
                last_eval_reward = eval_stats["avg_reward"]
                progress.set_description("training PPO")
                progress.set_postfix(
                    update=f"{update}/{end_update}",
                    reward=f"{avg_reward:.2f}",
                    length=f"{avg_length:.1f}",
                    macros=f"{avg_macros:.1f}",
                    macro_len=f"{avg_duration:.2f}",
                    loss=f"{stats.get('loss', 0.0):.3f}",
                    eval=f"{last_eval_reward:.2f}",
                )
                tqdm.write(
                    f"eval update={update:04d} "
                    f"avg_reward={eval_stats['avg_reward']:.3f} "
                    f"avg_length={eval_stats['avg_length']:.1f} "
                    f"avg_macros={eval_stats['avg_macros']:.1f}"
                )
                checkpoint_start = time.perf_counter()
                agent.save(checkpoint_path, update=update, eval_reward=eval_stats["avg_reward"])
                best_checkpoints = save_best_checkpoint(
                    agent,
                    checkpoint_dir,
                    update,
                    eval_stats["avg_reward"],
                    best_checkpoints,
                    args.keep_best_checkpoints,
                )
                checkpoint_sec = time.perf_counter() - checkpoint_start

            total_update_sec = time.perf_counter() - update_start
            episodes_per_sec = len(rewards) / rollout_sec if rollout_sec > 0 else 0.0
            metrics_writer.writerow(
                {
                    "update": update,
                    "episodes": len(rewards),
                    "workers": args.workers,
                    "worker_chunk_size": args.worker_chunk_size,
                    "device": args.device,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "avg_macros": avg_macros,
                    "avg_macro_duration": avg_duration,
                    "loss": stats.get("loss", 0.0),
                    "actor_loss": stats.get("actor_loss", 0.0),
                    "critic_loss": stats.get("critic_loss", 0.0),
                    "entropy": stats.get("entropy", 0.0),
                    "eval_avg_reward": "" if eval_stats is None else eval_stats["avg_reward"],
                    "eval_avg_length": "" if eval_stats is None else eval_stats["avg_length"],
                    "eval_avg_macros": "" if eval_stats is None else eval_stats["avg_macros"],
                    "rollout_sec": rollout_sec,
                    "ppo_sec": ppo_sec,
                    "eval_sec": eval_sec,
                    "checkpoint_sec": checkpoint_sec,
                    "total_update_sec": total_update_sec,
                    "episodes_per_sec": episodes_per_sec,
                    "rollout_pct": pct(rollout_sec, total_update_sec),
                    "ppo_pct": pct(ppo_sec, total_update_sec),
                    "eval_pct": pct(eval_sec, total_update_sec),
                    "checkpoint_pct": pct(checkpoint_sec, total_update_sec),
                    "recompute_logprob_sec": stats.get("recompute_logprob_sec", 0.0),
                    "critic_value_sec": stats.get("critic_value_sec", 0.0),
                    "num_macro_transitions": stats.get("num_macro_transitions", stats.get("transitions", 0)),
                    "avg_candidate_count": stats.get("avg_candidate_count", 0.0),
                    "min_candidate_count": stats.get("min_candidate_count", 0),
                    "max_candidate_count": stats.get("max_candidate_count", 0),
                    "recompute_batches": stats.get("recompute_batches", 0),
                    "avg_recompute_batch_size": stats.get("avg_recompute_batch_size", 0.0),
                    "decision_trigger_object_collected": stats.get("decision_trigger_object_collected", 0),
                    "decision_trigger_max_plan_steps": stats.get("decision_trigger_max_plan_steps", 0),
                    "remaining_objects_at_episode_end": stats.get("remaining_objects_at_episode_end", 0.0),
                }
            )
            metrics_handle.flush()
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)
        progress.close()
        metrics_handle.close()


if __name__ == "__main__":
    main()
