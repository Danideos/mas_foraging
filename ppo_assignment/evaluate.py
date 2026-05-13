import argparse
import random
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ppo_assignment.agent import AutoregressiveAssignmentPPOAgent
    from ppo_assignment.device import resolve_device
else:
    from .agent import AutoregressiveAssignmentPPOAgent
    from .device import resolve_device

try:
    from foraging.foraging import ForagingEnvironment
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from foraging import ForagingEnvironment


def run_episode(env, agent, render=False):
    state = env.reset()
    agent.reset_episode_state()
    total_reward = 0.0
    steps = 0
    macro_start = len(agent.rollout_buffer)

    while not env.done():
        actions = agent.action(state, deterministic=True)
        reward, world, agent_locations = env.perform_actions(actions)
        state = (world, agent_locations)
        agent.reward(reward)
        total_reward += reward
        steps += 1

    agent.finish_episode(state)
    macro_count = len(agent.rollout_buffer) - macro_start
    agent.rollout_buffer.clear()
    if render:
        env.render_history()
    return total_reward, steps, macro_count


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained foraging PPO checkpoint.")
    parser.add_argument("--checkpoint", default="foraging/ppo_assignment/checkpoints/ppo_assignment.pt")
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-plan-steps", type=int, default=None)
    parser.add_argument("--repeated-sync-penalty", type=float, default=0.0)
    parser.add_argument("--free-syncs-after-object", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--strict-device", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = str(resolve_device(args.device, strict=args.strict_device))

    agent = AutoregressiveAssignmentPPOAgent(
        w=args.width,
        h=args.height,
        agents=args.agents,
        hidden_dim=args.hidden_dim,
        max_plan_steps=args.max_plan_steps,
        repeated_sync_penalty=args.repeated_sync_penalty,
        free_syncs_after_object=args.free_syncs_after_object,
        device=args.device,
    )
    agent.load(args.checkpoint, load_optimizer=False)

    rewards = []
    lengths = []
    macros = []
    for episode in range(args.episodes):
        env = ForagingEnvironment(args.height, args.width, args.objects, args.agents)
        reward, length, macro_count = run_episode(
            env,
            agent,
            render=args.render and episode == 0,
        )
        rewards.append(reward)
        lengths.append(length)
        macros.append(macro_count)

    print(f"average reward: {sum(rewards) / len(rewards):.3f}")
    print(f"average episode length: {sum(lengths) / len(lengths):.1f}")
    print(f"average macro decisions: {sum(macros) / len(macros):.1f}")


if __name__ == "__main__":
    main()
