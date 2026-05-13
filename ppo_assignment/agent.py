import copy
import random
import time
from collections import defaultdict

import torch
from torch.distributions import Categorical

from .controller import (
    all_agents_completed_macro_goals,
    candidate_goals_for_agent,
    critic_goals,
    extract_objects,
    object_goals,
    goals_to_actions,
    manhattan,
    outward_wall_bump_action,
)
from .device import resolve_device
from .models import Actor, Critic
from .rollout import MacroTransition, RolloutBuffer


class AutoregressiveAssignmentPPOAgent:
    def __init__(
        self,
        w,
        h,
        agents,
        device="cpu",
        hidden_dim=128,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        max_plan_steps=None,
    ):
        self.w = w
        self.h = h
        self.agents = agents
        self.device = resolve_device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.max_plan_steps = max_plan_steps if max_plan_steps is not None else max(h + w, 10)

        self.actor = Actor(hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.rollout_buffer = RolloutBuffer()
        self.reset_rollout_metrics()
        self.reset_episode_state()
        self.last_update_stats = {}

    def reset_rollout_metrics(self):
        self.rollout_metrics = {
            "sync_choice_count": 0,
            "zero_distance_sync_choice_count": 0,
            "goal_choice_count": 0,
            "sync_bump_completed_count": 0,
            "sync_reached_but_not_bumped_count": 0,
            "decision_trigger_object_collected": 0,
            "decision_trigger_goals_completed": 0,
            "decision_trigger_max_plan_steps": 0,
            "remaining_objects_at_episode_end_sum": 0,
            "episode_end_count": 0,
        }

    def get_rollout_metrics(self):
        return dict(self.rollout_metrics)

    def reset_episode_state(self):
        self.active_plan = None
        self.plan_duration = 0
        self.last_object_count = None
        self.current_macro = None
        self.sync_bumped = {}
        self.pending_sync_bump_agents = set()

    def action(self, state, deterministic=False):
        world_map, agent_locations = state
        objects = extract_objects(world_map)
        current_object_count = len(objects)

        if current_object_count == 0:
            self.last_object_count = current_object_count
            return [random.randrange(0, 4) for _ in agent_locations]

        object_collected = (
            self.last_object_count is not None
            and current_object_count < self.last_object_count
        )
        goals_completed = (
            self.active_plan is not None
            and current_object_count > 0
            and all_agents_completed_macro_goals(agent_locations, self.active_plan, self.sync_bumped)
        )
        max_plan_steps_reached = self.active_plan is not None and self.plan_duration >= self.max_plan_steps
        need_decision = (
            self.active_plan is None
            or object_collected
            or goals_completed
            or max_plan_steps_reached
        )

        if need_decision:
            if object_collected:
                self.rollout_metrics["decision_trigger_object_collected"] += 1
            if goals_completed:
                self.rollout_metrics["decision_trigger_goals_completed"] += 1
            if max_plan_steps_reached:
                self.rollout_metrics["decision_trigger_max_plan_steps"] += 1

            if self.current_macro is not None:
                self.close_current_macro(next_state=state, done=False)

            with torch.no_grad():
                active_plan, target_indices, agent_order, joint_logprob, joint_entropy = self.make_assignment_decision(
                    state,
                    deterministic=deterministic,
                )
                goals = critic_goals(world_map, agent_locations, self.h, self.w)
                value = self.critic(state, self.h, self.w, self.agents, goals)

            self.active_plan = active_plan
            self.rollout_metrics["goal_choice_count"] += len(active_plan)
            for agent_idx, goal in active_plan.items():
                if goal[0] == "sync":
                    self.rollout_metrics["sync_choice_count"] += 1
                    if manhattan(agent_locations[agent_idx], goal) == 0:
                        self.rollout_metrics["zero_distance_sync_choice_count"] += 1
            self.sync_bumped = {
                agent_idx: False
                for agent_idx, goal in active_plan.items()
                if goal[0] == "sync"
            }
            self.pending_sync_bump_agents = set()
            self.current_macro = {
                "decision_state": self.copy_state(state),
                "agent_order": list(agent_order),
                "targets": copy.deepcopy(active_plan),
                "target_indices": copy.deepcopy(target_indices),
                "old_joint_logprob": float(joint_logprob.detach().cpu().item()),
                "old_value": float(value.detach().cpu().item()),
                "old_entropy": float(joint_entropy.detach().cpu().item()),
                "macro_reward": 0.0,
                "duration": 0,
            }
            self.plan_duration = 0

        primitive_actions = self.goals_to_primitive_actions(agent_locations)
        self.last_object_count = current_object_count
        return primitive_actions

    def goals_to_primitive_actions(self, agent_locations):
        actions = []
        self.pending_sync_bump_agents.clear()
        for agent_idx, agent_pos in enumerate(agent_locations):
            goal = self.active_plan.get(agent_idx) if self.active_plan else None
            if goal is None:
                actions.append(random.randrange(0, 4))
                continue

            if (
                goal[0] == "sync"
                and not self.sync_bumped.get(agent_idx, False)
                and manhattan(agent_pos, goal) == 0
            ):
                actions.append(outward_wall_bump_action(goal, self.h, self.w))
                self.pending_sync_bump_agents.add(agent_idx)
                self.rollout_metrics["sync_reached_but_not_bumped_count"] += 1
            else:
                actions.append(goals_to_actions([agent_pos], {0: goal}, self.h, self.w)[0])
        return actions

    def make_assignment_decision(self, state, deterministic=False):
        world_map, agent_locations = state
        encoder_goals = critic_goals(world_map, agent_locations, self.h, self.w)
        if not encoder_goals:
            return {}, {}, list(range(len(agent_locations))), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_state(
            state,
            self.h,
            self.w,
            self.agents,
            encoder_goals,
        )
        agent_order = list(range(len(agent_locations)))
        if deterministic:
            agent_order.sort()
        else:
            random.shuffle(agent_order)

        previous_decision_tokens = []
        targets = {}
        target_indices = {}
        joint_logprob = torch.tensor(0.0, device=self.device)
        joint_entropy = torch.tensor(0.0, device=self.device)

        for step_idx, agent_idx in enumerate(agent_order):
            candidate_goals = candidate_goals_for_agent(world_map, agent_locations[agent_idx], self.h, self.w)
            candidate_goal_indices = [encoder_goals.index(goal) for goal in candidate_goals]
            logits = self.actor.decision_logits(
                agent_idx,
                step_idx,
                previous_decision_tokens,
                agent_locations,
                candidate_goals,
                agent_embeddings,
                goal_embeddings[candidate_goal_indices],
                global_embedding,
                self.h,
                self.w,
                self.agents,
            )
            dist = Categorical(logits=logits)
            if deterministic:
                selected_goal_idx = torch.argmax(logits)
            else:
                selected_goal_idx = dist.sample()
            goal_idx_int = int(selected_goal_idx.item())
            targets[agent_idx] = candidate_goals[goal_idx_int]
            target_indices[agent_idx] = goal_idx_int
            joint_logprob = joint_logprob + dist.log_prob(selected_goal_idx)
            joint_entropy = joint_entropy + dist.entropy()
            previous_decision_tokens.append(
                self.actor.build_previous_decision_token(
                    agent_idx,
                    candidate_goal_indices[goal_idx_int],
                    step_idx,
                    agent_locations,
                    encoder_goals,
                    agent_embeddings,
                    goal_embeddings,
                    self.h,
                    self.w,
                    self.agents,
                )
            )

        return targets, target_indices, agent_order, joint_logprob, joint_entropy

    def reward(self, reward):
        for agent_idx in self.pending_sync_bump_agents:
            if agent_idx in self.sync_bumped and not self.sync_bumped[agent_idx]:
                self.sync_bumped[agent_idx] = True
                self.rollout_metrics["sync_bump_completed_count"] += 1
        self.pending_sync_bump_agents.clear()

        if self.current_macro is not None:
            duration = self.current_macro["duration"]
            self.current_macro["macro_reward"] += (self.gamma ** duration) * float(reward)
            self.current_macro["duration"] += 1
        self.plan_duration += 1

    def finish_episode(self, final_state):
        if self.current_macro is not None:
            self.close_current_macro(next_state=final_state, done=True)
        self.rollout_metrics["remaining_objects_at_episode_end_sum"] += len(extract_objects(final_state[0]))
        self.rollout_metrics["episode_end_count"] += 1
        self.active_plan = None
        self.current_macro = None
        self.sync_bumped = {}
        self.pending_sync_bump_agents = set()

    def close_current_macro(self, next_state, done):
        macro = self.current_macro
        self.rollout_buffer.add(
            MacroTransition(
                decision_state=macro["decision_state"],
                agent_order=list(macro["agent_order"]),
                targets=copy.deepcopy(macro["targets"]),
                target_indices=copy.deepcopy(macro["target_indices"]),
                old_joint_logprob=macro["old_joint_logprob"],
                old_value=macro["old_value"],
                old_entropy=macro["old_entropy"],
                macro_reward=float(macro["macro_reward"]),
                duration=int(macro["duration"]),
                next_decision_state=self.copy_state(next_state),
                done=bool(done),
            )
        )
        self.current_macro = None

    def recompute_logprob(self, decision_state, agent_order, targets, target_indices=None):
        world_map, agent_locations = decision_state
        encoder_goals = critic_goals(world_map, agent_locations, self.h, self.w)
        if not encoder_goals:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_state(
            decision_state,
            self.h,
            self.w,
            self.agents,
            encoder_goals,
        )
        previous_decision_tokens = []
        new_joint_logprob = torch.tensor(0.0, device=self.device)
        new_entropy = torch.tensor(0.0, device=self.device)

        for step_idx, agent_idx in enumerate(agent_order):
            stored_target = targets[agent_idx]
            candidate_goals = candidate_goals_for_agent(world_map, agent_locations[agent_idx], self.h, self.w)
            candidate_goal_indices = [encoder_goals.index(goal) for goal in candidate_goals]
            if target_indices is None:
                try:
                    stored_candidate_goal_idx = candidate_goals.index(stored_target)
                except ValueError as exc:
                    raise ValueError(f"Stored goal {stored_target} is missing from agent candidate goals.") from exc
            else:
                stored_candidate_goal_idx = target_indices[agent_idx]
                if candidate_goals[stored_candidate_goal_idx] != stored_target:
                    raise ValueError(
                        f"Stored candidate index {stored_candidate_goal_idx} maps to "
                        f"{candidate_goals[stored_candidate_goal_idx]}, expected {stored_target}."
                    )

            logits = self.actor.decision_logits(
                agent_idx,
                step_idx,
                previous_decision_tokens,
                agent_locations,
                candidate_goals,
                agent_embeddings,
                goal_embeddings[candidate_goal_indices],
                global_embedding,
                self.h,
                self.w,
                self.agents,
            )
            dist = Categorical(logits=logits)
            action_tensor = torch.tensor(stored_candidate_goal_idx, dtype=torch.long, device=self.device)
            new_joint_logprob = new_joint_logprob + dist.log_prob(action_tensor)
            new_entropy = new_entropy + dist.entropy()
            previous_decision_tokens.append(
                self.actor.build_previous_decision_token(
                    agent_idx,
                    candidate_goal_indices[stored_candidate_goal_idx],
                    step_idx,
                    agent_locations,
                    encoder_goals,
                    agent_embeddings,
                    goal_embeddings,
                    self.h,
                    self.w,
                    self.agents,
                )
            )

        return new_joint_logprob, new_entropy

    def recompute_logprob_batch(self, transitions):
        states = [transition.decision_state for transition in transitions]
        encoder_goals_batch = [critic_goals(state[0], state[1], self.h, self.w) for state in states]
        if not encoder_goals_batch or len(encoder_goals_batch[0]) == 0:
            zeros = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
            return zeros, zeros

        agent_locations_batch = [state[1] for state in states]
        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_states_batch(
            states,
            self.h,
            self.w,
            self.agents,
            encoder_goals_batch,
        )

        previous_decision_tokens = []
        batch_size = len(transitions)
        new_joint_logprob = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        new_entropy = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for step_idx in range(self.agents):
            agent_indices_all = [transition.agent_order[step_idx] for transition in transitions]
            stored_candidate_goal_indices_all = []
            candidate_goals_batch_all = []
            candidate_encoder_indices_batch_all = []
            for row, transition in enumerate(transitions):
                stored_target = transition.targets[agent_indices_all[row]]
                candidate_goals = candidate_goals_for_agent(
                    states[row][0],
                    agent_locations_batch[row][agent_indices_all[row]],
                    self.h,
                    self.w,
                )
                candidate_encoder_indices = [encoder_goals_batch[row].index(goal) for goal in candidate_goals]
                candidate_goals_batch_all.append(candidate_goals)
                candidate_encoder_indices_batch_all.append(candidate_encoder_indices)
                stored_candidate_goal_idx = transition.target_indices[agent_indices_all[row]]
                if candidate_goals[stored_candidate_goal_idx] != stored_target:
                    raise ValueError(
                        f"Stored candidate index {stored_candidate_goal_idx} maps to "
                        f"{candidate_goals[stored_candidate_goal_idx]}, expected {stored_target}."
                    )
                stored_candidate_goal_indices_all.append(stored_candidate_goal_idx)

            selected_encoder_goal_indices = [None] * batch_size
            candidate_goal_embeddings = torch.stack(
                [
                    goal_embeddings[row, torch.tensor(candidate_encoder_indices_batch_all[row], dtype=torch.long, device=self.device)]
                    for row in range(batch_size)
                ],
                dim=0,
            )
            logits = self.actor.decision_logits_batch(
                agent_indices_all,
                step_idx,
                previous_decision_tokens,
                agent_locations_batch,
                candidate_goals_batch_all,
                agent_embeddings,
                candidate_goal_embeddings,
                global_embedding,
                self.h,
                self.w,
                self.agents,
            )
            dist = Categorical(logits=logits)
            action_tensor = torch.tensor(stored_candidate_goal_indices_all, dtype=torch.long, device=self.device)
            new_joint_logprob = new_joint_logprob + dist.log_prob(action_tensor)
            new_entropy = new_entropy + dist.entropy()
            for row in range(batch_size):
                selected_encoder_goal_indices[row] = candidate_encoder_indices_batch_all[row][stored_candidate_goal_indices_all[row]]

            token_batch = self.actor.build_previous_decision_tokens_batch(
                agent_indices_all,
                selected_encoder_goal_indices,
                step_idx,
                agent_locations_batch,
                encoder_goals_batch,
                agent_embeddings,
                goal_embeddings,
                self.h,
                self.w,
                self.agents,
            )
            previous_decision_tokens.append(token_batch)

        return new_joint_logprob, new_entropy

    def critic_values_batch(self, states):
        if not states:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        values = [None] * len(states)
        groups = defaultdict(list)
        goals_cache = []
        for idx, state in enumerate(states):
            goals = critic_goals(state[0], state[1], self.h, self.w)
            goals_cache.append(goals)
            groups[len(goals)].append(idx)

        for indices in groups.values():
            group_states = [states[idx] for idx in indices]
            group_goals = [goals_cache[idx] for idx in indices]
            group_values = self.critic.forward_batch(
                group_states,
                self.h,
                self.w,
                self.agents,
                group_goals,
            )
            for offset, original_idx in enumerate(indices):
                values[original_idx] = group_values[offset]

        return torch.stack(values)

    def recompute_logprob_all_batched(self, transitions):
        logprobs = [None] * len(transitions)
        entropies = [None] * len(transitions)
        groups = defaultdict(list)
        for idx, transition in enumerate(transitions):
            groups[len(object_goals(transition.decision_state[0]))].append(idx)

        self._last_recompute_batches = len(groups)
        self._last_avg_recompute_batch_size = len(transitions) / max(len(groups), 1)
        for indices in groups.values():
            group_transitions = [transitions[idx] for idx in indices]
            group_logprobs, group_entropies = self.recompute_logprob_batch(group_transitions)
            for offset, original_idx in enumerate(indices):
                logprobs[original_idx] = group_logprobs[offset]
                entropies[original_idx] = group_entropies[offset]

        return torch.stack(logprobs), torch.stack(entropies)

    def train_update(self):
        if len(self.rollout_buffer) == 0:
            return {"transitions": 0}

        transitions = list(self.rollout_buffer)
        self._last_recompute_batches = 0
        self._last_avg_recompute_batch_size = 0.0
        critic_value_sec = 0.0
        recompute_logprob_sec = 0.0
        with torch.no_grad():
            critic_start = time.perf_counter()
            next_values = self.critic_values_batch([transition.next_decision_state for transition in transitions])
            critic_value_sec += time.perf_counter() - critic_start
            macro_rewards = torch.tensor(
                [transition.macro_reward for transition in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            durations = torch.tensor(
                [transition.duration for transition in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            not_done = torch.tensor(
                [0.0 if transition.done else 1.0 for transition in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            old_values = torch.tensor(
                [
                    float(transition.old_value)
                    if not isinstance(transition.old_value, torch.Tensor)
                    else float(transition.old_value.detach().cpu().item())
                    for transition in transitions
                ],
                dtype=torch.float32,
                device=self.device,
            )
            target_tensor = (
                macro_rewards + torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), durations) * next_values * not_done
            ).detach()
            advantage_tensor = (target_tensor - old_values).detach()
            if len(advantage_tensor) > 1:
                advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)

        stats = {}
        for _ in range(self.ppo_epochs):
            recompute_start = time.perf_counter()
            new_logprob_tensor, new_entropy_tensor = self.recompute_logprob_all_batched(transitions)
            recompute_logprob_sec += time.perf_counter() - recompute_start
            critic_start = time.perf_counter()
            new_value_tensor = self.critic_values_batch([transition.decision_state for transition in transitions])
            critic_value_sec += time.perf_counter() - critic_start
            old_logprob_tensor = torch.tensor(
                [
                    float(transition.old_joint_logprob)
                    if not isinstance(transition.old_joint_logprob, torch.Tensor)
                    else float(transition.old_joint_logprob.detach().cpu().item())
                    for transition in transitions
                ],
                dtype=torch.float32,
                device=self.device,
            )

            ratio = torch.exp(new_logprob_tensor - old_logprob_tensor)
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_tensor
            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = ((new_value_tensor - target_tensor) ** 2).mean()
            entropy_bonus = new_entropy_tensor.mean()
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            stats = {
                "transitions": len(transitions),
                "loss": float(loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "entropy": float(entropy_bonus.detach().cpu()),
                "avg_macro_duration": sum(t.duration for t in transitions) / len(transitions),
                "recompute_logprob_sec": recompute_logprob_sec,
                "critic_value_sec": critic_value_sec,
                "num_macro_transitions": len(transitions),
                "avg_candidate_count": sum(len(object_goals(t.decision_state[0])) + 4 for t in transitions) / len(transitions),
                "min_candidate_count": min(len(object_goals(t.decision_state[0])) + 4 for t in transitions),
                "max_candidate_count": max(len(object_goals(t.decision_state[0])) + 4 for t in transitions),
                "recompute_batches": self._last_recompute_batches,
                "avg_recompute_batch_size": self._last_avg_recompute_batch_size,
            }

        self.rollout_buffer.clear()
        self.last_update_stats = stats
        return stats

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {
                    "w": self.w,
                    "h": self.h,
                    "agents": self.agents,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_eps": self.clip_eps,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "ppo_epochs": self.ppo_epochs,
                    "max_plan_steps": self.max_plan_steps,
                },
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @staticmethod
    def copy_state(state):
        world_map, agent_locations = state
        return copy.deepcopy(world_map), copy.deepcopy(agent_locations)
