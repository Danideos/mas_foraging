import copy
import random
import time
from collections import defaultdict

import torch
from torch.distributions import Categorical

from .controller import (
    candidate_goals_for_agent,
    critic_goals,
    extract_objects,
    legal_pace_action,
    object_goals,
    manhattan,
    move_towards,
    outward_wall_bump_action,
    target_exists,
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
        ppo_minibatch_size=256,
        ppo_token_budget=8192,
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
        self.ppo_minibatch_size = ppo_minibatch_size
        self.ppo_token_budget = ppo_token_budget

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

    def model_batch_size(self, num_agents, num_goals):
        max_batch_size = max(int(self.ppo_minibatch_size), 1)
        token_budget = int(self.ppo_token_budget or 0)
        if token_budget <= 0:
            return max_batch_size
        tokens_per_transition = max(int(num_agents) + int(num_goals), 1)
        return max(1, min(max_batch_size, token_budget // tokens_per_transition))

    def batch_slices(self, length, batch_size=None):
        if batch_size is None:
            batch_size = self.ppo_minibatch_size
        batch_size = max(int(batch_size), 1)
        for start in range(0, length, batch_size):
            yield start, min(start + batch_size, length)

    def index_subset(self, values, indices):
        return [values[idx] for idx in indices]

    def transition_token_count(self, transition):
        full_agent_count = len(transition.decision_state[1])
        return max(full_agent_count + self.transition_candidate_count(transition), 1)

    def dynamic_transition_batches(self, indices, transitions):
        max_batch_size = max(int(self.ppo_minibatch_size), 1)
        token_budget = int(self.ppo_token_budget or 0)
        if token_budget <= 0:
            for start, end in self.batch_slices(len(indices), max_batch_size):
                yield indices[start:end]
            return

        current = []
        current_tokens = 0
        for idx in indices:
            transition_tokens = self.transition_token_count(transitions[idx])
            if current and (
                len(current) >= max_batch_size
                or current_tokens + transition_tokens > token_budget
            ):
                yield current
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += transition_tokens
        if current:
            yield current

    def reset_rollout_metrics(self):
        self.rollout_metrics = {
            "goal_choice_count": 0,
            "decision_trigger_object_collected": 0,
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
        self.object_bounce_targets = {}
        self.parity_phase_done = False
        self.parity_phase = None
        self._last_world_map = None

    def nearest_edge_goal(self, agent_pos, h, w):
        x, y = agent_pos
        candidates = [
            (x, ("edge", 0, y, 0)),
            (h - 1 - x, ("edge", h - 1, y, 0)),
            (y, ("edge", x, 0, 0)),
            (w - 1 - y, ("edge", x, w - 1, 0)),
        ]
        _, goal = min(candidates, key=lambda item: item[0])
        return goal

    def initialize_parity_phase(self, agent_locations, h, w):
        parity_groups = {0: [], 1: []}
        for agent_idx, (x, y) in enumerate(agent_locations):
            parity_groups[(x + y) % 2].append(agent_idx)

        if not parity_groups[0] or not parity_groups[1]:
            self.parity_phase_done = True
            self.parity_phase = None
            return

        group_scores = []
        for parity, agent_indices in parity_groups.items():
            bump_costs = []
            for agent_idx in agent_indices:
                x, y = agent_locations[agent_idx]
                nearest_edge_distance = min(x, h - 1 - x, y, w - 1 - y)
                bump_costs.append(nearest_edge_distance + 1)
            group_scores.append(
                (
                    max(bump_costs),
                    sum(bump_costs),
                    len(agent_indices),
                    parity,
                )
            )

        _, _, _, selected_parity = min(group_scores)
        selected_agents = set(parity_groups[selected_parity])
        ready_agents = set(parity_groups[1 - selected_parity])
        self.parity_phase = {
            "selected_agents": selected_agents,
            "ready_agents": ready_agents,
            "edge_goals": {
                agent_idx: self.nearest_edge_goal(agent_locations[agent_idx], h, w)
                for agent_idx in selected_agents
            },
            "completed_agents": set(),
        }

    def parity_sync_complete(self):
        if self.parity_phase is None:
            return self.parity_phase_done
        return self.parity_phase["completed_agents"] == self.parity_phase["selected_agents"]

    def parity_sync_actions(self, agent_locations, h, w):
        if self.parity_phase is None:
            self.initialize_parity_phase(agent_locations, h, w)

        if self.parity_phase_done:
            return {}

        selected_agents = self.parity_phase["selected_agents"]
        completed_agents = self.parity_phase["completed_agents"]
        edge_goals = self.parity_phase["edge_goals"]

        actions = {}
        for agent_idx, agent_pos in enumerate(agent_locations):
            if agent_idx not in selected_agents:
                continue
            if agent_idx in completed_agents:
                actions[agent_idx] = legal_pace_action(agent_pos, h, w)
                continue

            edge_goal = edge_goals[agent_idx]
            if agent_pos[0] == edge_goal[1] and agent_pos[1] == edge_goal[2]:
                actions[agent_idx] = outward_wall_bump_action(edge_goal, h, w)
                completed_agents.add(agent_idx)
            else:
                actions[agent_idx] = move_towards(agent_pos, edge_goal, h, w)

        return actions

    def parity_ready_indices(self):
        if self.parity_phase is None:
            return []
        return sorted(self.parity_phase["ready_agents"])

    @staticmethod
    def filter_object_goals(world_map, max_level):
        return [goal for goal in object_goals(world_map) if goal[3] <= max_level]

    @staticmethod
    def visible_state(state, visible_agent_indices):
        world_map, agent_locations = state
        return world_map, [agent_locations[agent_idx] for agent_idx in visible_agent_indices]

    def active_plan_copy(self):
        return {} if self.active_plan is None else copy.deepcopy(self.active_plan)

    @staticmethod
    def valid_object_assignment(world_map, goal):
        return goal is not None and goal[0] == "object" and target_exists(world_map, goal[1:])

    def visible_agents_for_decision(self, world_map, agent_count, extra_visible=None):
        extra_visible = set() if extra_visible is None else set(extra_visible)
        active_plan = self.active_plan or {}
        visible = []
        for agent_idx in range(agent_count):
            goal = active_plan.get(agent_idx)
            if agent_idx in extra_visible or not self.valid_object_assignment(world_map, goal):
                visible.append(agent_idx)
        return visible

    def assignment_counts(self, goals, active_plan_before, visible_agent_indices, visible_targets=None):
        visible_set = set(visible_agent_indices or [])
        visible_targets = {} if visible_targets is None else visible_targets
        locked_counts = {goal: 0 for goal in goals}
        visible_counts = {goal: 0 for goal in goals}

        for agent_idx, goal in (active_plan_before or {}).items():
            if goal in locked_counts and agent_idx not in visible_set:
                locked_counts[goal] += 1
        for goal in visible_targets.values():
            if goal in visible_counts:
                visible_counts[goal] += 1
        return locked_counts, visible_counts

    def assignment_context(self, agent_pos, candidate_goals, active_plan_before, visible_agent_indices, visible_targets, h, w):
        distance_scale = max(float((h - 1) + (w - 1)), 1.0)
        agent_scale = max(float(len(active_plan_before or {}) or self.agents), 1.0)
        locked_counts, visible_counts = self.assignment_counts(
            candidate_goals,
            active_plan_before,
            visible_agent_indices,
            visible_targets=visible_targets,
        )
        rows = []
        ax, ay = agent_pos
        for goal in candidate_goals:
            level = float(goal[3])
            locked_count = float(locked_counts.get(goal, 0))
            visible_count = float(visible_counts.get(goal, 0))
            total_count = locked_count + visible_count
            total_after = total_count + 1.0
            remaining_need = max(0.0, level - total_count)
            rows.append(
                [
                    (abs(float(ax) - float(goal[1])) + abs(float(ay) - float(goal[2]))) / distance_scale,
                    level / max(float(self.agents), 1.0),
                    locked_count / agent_scale,
                    visible_count / agent_scale,
                    total_count / agent_scale,
                    remaining_need / agent_scale,
                    1.0 if total_after >= level else 0.0,
                    1.0 if total_after > level else 0.0,
                ]
            )
        return rows

    def action(self, state, deterministic=False):
        world_map, agent_locations = state
        h = len(world_map)
        w = len(world_map[0]) if h > 0 else self.w
        num_agents = len(agent_locations)
        objects = extract_objects(world_map)
        current_object_count = len(objects)

        if current_object_count == 0:
            self.last_object_count = current_object_count
            return [legal_pace_action(agent_pos, h, w) for agent_pos in agent_locations]

        object_collected = (
            self.last_object_count is not None
            and current_object_count < self.last_object_count
        )

        if not self.parity_phase_done:
            return self.parity_phase_actions(
                state,
                deterministic=deterministic,
                object_collected=object_collected,
                current_object_count=current_object_count,
            )

        max_plan_steps_reached = self.active_plan is not None and self.plan_duration >= self.max_plan_steps
        need_decision = (
            self.active_plan is None
            or object_collected
            or max_plan_steps_reached
        )

        if need_decision:
            active_plan_before = self.active_plan_copy()
            extra_visible = self.current_macro.get("visible_agent_indices", []) if max_plan_steps_reached and self.current_macro else []
            visible_agent_indices = self.visible_agents_for_decision(world_map, num_agents, extra_visible=extra_visible)
            if object_collected:
                self.rollout_metrics["decision_trigger_object_collected"] += 1
            if max_plan_steps_reached:
                self.rollout_metrics["decision_trigger_max_plan_steps"] += 1

            if self.current_macro is not None:
                self.close_current_macro(next_state=state, done=False)

            self.active_plan = active_plan_before
            self.object_bounce_targets = {
                agent_idx: goal
                for agent_idx, goal in self.object_bounce_targets.items()
                if self.valid_object_assignment(world_map, self.active_plan.get(agent_idx))
            }

            goals = critic_goals(world_map, agent_locations, h, w)
            if visible_agent_indices and goals:
                with torch.no_grad():
                    targets, target_indices, agent_order, joint_logprob, joint_entropy = self.make_assignment_decision(
                        state,
                        deterministic=deterministic,
                        visible_agent_indices=visible_agent_indices,
                        active_plan_before=active_plan_before,
                    )
                    value = self.critic(state, h, w, num_agents, goals)

                for agent_idx, goal in targets.items():
                    self.active_plan[agent_idx] = goal
                self.rollout_metrics["goal_choice_count"] += len(targets)
                self.current_macro = {
                    "mode": "normal",
                    "decision_state": self.copy_state(state),
                    "active_plan_before": copy.deepcopy(active_plan_before),
                    "active_plan_after": self.active_plan_copy(),
                    "visible_agent_indices": list(visible_agent_indices),
                    "filtered_goals": None,
                    "agent_order": list(agent_order),
                    "targets": copy.deepcopy(targets),
                    "target_indices": copy.deepcopy(target_indices),
                    "old_joint_logprob": float(joint_logprob.detach().cpu().item()),
                    "old_value": float(value.detach().cpu().item()),
                    "old_entropy": float(joint_entropy.detach().cpu().item()),
                    "macro_reward": 0.0,
                    "duration": 0,
                }
                self.plan_duration = 0

        self._last_world_map = world_map
        primitive_actions = self.goals_to_primitive_actions(agent_locations, h, w)
        self.last_object_count = current_object_count
        return primitive_actions

    def parity_phase_actions(self, state, deterministic, object_collected, current_object_count):
        world_map, agent_locations = state
        h = len(world_map)
        w = len(world_map[0]) if h > 0 else self.w
        if self.parity_phase is None:
            self.initialize_parity_phase(agent_locations, h, w)

        if self.parity_phase_done:
            return self.action(state, deterministic=deterministic)

        sync_complete = self.parity_sync_complete()
        max_plan_steps_reached = self.current_macro is not None and self.plan_duration >= self.max_plan_steps
        timed_out_visible = self.current_macro.get("visible_agent_indices", []) if max_plan_steps_reached and self.current_macro else []
        if self.current_macro is not None and (
            object_collected or sync_complete or max_plan_steps_reached
        ):
            if sync_complete:
                self.parity_phase_done = True
                self.parity_phase = None
            if object_collected:
                self.rollout_metrics["decision_trigger_object_collected"] += 1
            if max_plan_steps_reached:
                self.rollout_metrics["decision_trigger_max_plan_steps"] += 1
            self.close_current_macro(next_state=state, done=False)
            self.object_bounce_targets = {
                agent_idx: goal
                for agent_idx, goal in self.object_bounce_targets.items()
                if self.active_plan and self.valid_object_assignment(world_map, self.active_plan.get(agent_idx))
            }

        if sync_complete:
            self.parity_phase_done = True
            self.parity_phase = None
            return self.action(state, deterministic=deterministic)

        sync_actions = self.parity_sync_actions(agent_locations, h, w)
        ready_indices = self.parity_ready_indices()
        filtered_goals = self.filter_object_goals(world_map, len(ready_indices))

        active_plan_before = self.active_plan_copy()
        visible_ready_indices = [
            agent_idx
            for agent_idx in ready_indices
            if agent_idx in timed_out_visible or not self.valid_object_assignment(world_map, active_plan_before.get(agent_idx))
        ]

        if self.current_macro is None and visible_ready_indices and filtered_goals:
            with torch.no_grad():
                targets, target_indices, agent_order, joint_logprob, joint_entropy = self.make_assignment_decision(
                    state,
                    deterministic=deterministic,
                    visible_agent_indices=visible_ready_indices,
                    filtered_goals=filtered_goals,
                    active_plan_before=active_plan_before,
                )
                goals = critic_goals(world_map, agent_locations, h, w)
                value = self.critic(state, h, w, len(agent_locations), goals)

            self.active_plan = active_plan_before
            for agent_idx, goal in targets.items():
                self.active_plan[agent_idx] = goal
            self.rollout_metrics["goal_choice_count"] += len(targets)
            self.current_macro = {
                "mode": "parity_ready_subproblem",
                "decision_state": self.copy_state(state),
                "active_plan_before": copy.deepcopy(active_plan_before),
                "active_plan_after": self.active_plan_copy(),
                "visible_agent_indices": list(visible_ready_indices),
                "filtered_goals": copy.deepcopy(filtered_goals),
                "agent_order": list(agent_order),
                "targets": copy.deepcopy(targets),
                "target_indices": copy.deepcopy(target_indices),
                "old_joint_logprob": float(joint_logprob.detach().cpu().item()),
                "old_value": float(value.detach().cpu().item()),
                "old_entropy": float(joint_entropy.detach().cpu().item()),
                "macro_reward": 0.0,
                "duration": 0,
            }
            self.plan_duration = 0

        self._last_world_map = world_map
        ready_actions = self.goals_to_primitive_actions(agent_locations, h, w)
        actions = []
        for agent_idx, agent_pos in enumerate(agent_locations):
            if agent_idx in sync_actions:
                actions.append(sync_actions[agent_idx])
            elif agent_idx in ready_indices:
                actions.append(ready_actions[agent_idx])
            else:
                actions.append(legal_pace_action(agent_pos, h, w))

        self.last_object_count = current_object_count
        return actions

    def goals_to_primitive_actions(self, agent_locations, h=None, w=None):
        if h is None:
            h = self.h
        if w is None:
            w = self.w
        actions = []
        for agent_idx, agent_pos in enumerate(agent_locations):
            goal = self.active_plan.get(agent_idx) if self.active_plan else None
            if goal is None:
                actions.append(legal_pace_action(agent_pos, h, w))
                continue

            if agent_idx in self.object_bounce_targets:
                bounce_goal = self.object_bounce_targets.pop(agent_idx)
                actions.append(move_towards(agent_pos, bounce_goal, h, w))
            elif target_exists(self._last_world_map, goal[1:]) and manhattan(agent_pos, goal) == 0:
                action = legal_pace_action(agent_pos, h, w)
                actions.append(action)
                self.object_bounce_targets[agent_idx] = goal
            else:
                actions.append(move_towards(agent_pos, goal, h, w))
        return actions

    def make_assignment_decision(
        self,
        state,
        deterministic=False,
        visible_agent_indices=None,
        filtered_goals=None,
        active_plan_before=None,
    ):
        world_map, agent_locations = state
        h = len(world_map)
        w = len(world_map[0]) if h > 0 else self.w
        if visible_agent_indices is None:
            visible_agent_indices = list(range(len(agent_locations)))
        else:
            visible_agent_indices = list(visible_agent_indices)
        active_plan_before = {} if active_plan_before is None else copy.deepcopy(active_plan_before)
        num_agents = len(agent_locations)
        encoder_goals = (
            list(filtered_goals)
            if filtered_goals is not None
            else critic_goals(world_map, agent_locations, h, w)
        )
        if not encoder_goals:
            return {}, {}, list(visible_agent_indices), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_state(
            state,
            h,
            w,
            num_agents,
            encoder_goals,
        )
        agent_order = list(visible_agent_indices)
        if deterministic:
            agent_order.sort()
        else:
            random.shuffle(agent_order)

        previous_decision_tokens = []
        targets = {}
        target_indices = {}
        joint_logprob = torch.tensor(0.0, device=self.device)
        joint_entropy = torch.tensor(0.0, device=self.device)
        visible_targets_so_far = {}

        for step_idx, agent_idx in enumerate(agent_order):
            candidate_goals = (
                list(filtered_goals)
                if filtered_goals is not None
                else candidate_goals_for_agent(world_map, agent_locations[agent_idx], h, w)
            )
            candidate_goal_indices = [encoder_goals.index(goal) for goal in candidate_goals]
            assignment_context = self.assignment_context(
                agent_locations[agent_idx],
                candidate_goals,
                active_plan_before,
                visible_agent_indices,
                visible_targets_so_far,
                h,
                w,
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
                h,
                w,
                num_agents,
                assignment_context=assignment_context,
            )
            dist = Categorical(logits=logits)
            if deterministic:
                selected_goal_idx = torch.argmax(logits)
            else:
                selected_goal_idx = dist.sample()
            goal_idx_int = int(selected_goal_idx.item())
            targets[agent_idx] = candidate_goals[goal_idx_int]
            visible_targets_so_far[agent_idx] = candidate_goals[goal_idx_int]
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
                    h,
                    w,
                    num_agents,
                )
            )

        return targets, target_indices, agent_order, joint_logprob, joint_entropy

    def reward(self, reward):
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
        self.object_bounce_targets = {}

    def close_current_macro(self, next_state, done):
        macro = self.current_macro
        next_visible_agent_indices = None
        next_filtered_goals = None
        if (
            not done
            and macro.get("mode") == "parity_ready_subproblem"
            and not self.parity_phase_done
            and self.parity_phase is not None
        ):
            next_visible_agent_indices = self.parity_ready_indices()
            next_filtered_goals = self.filter_object_goals(next_state[0], len(next_visible_agent_indices))
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
                mode=macro.get("mode", "normal"),
                active_plan_before=copy.deepcopy(macro.get("active_plan_before")),
                active_plan_after=copy.deepcopy(self.active_plan if self.active_plan is not None else macro.get("active_plan_after")),
                visible_agent_indices=copy.deepcopy(macro.get("visible_agent_indices")),
                filtered_goals=copy.deepcopy(macro.get("filtered_goals")),
                next_visible_agent_indices=copy.deepcopy(next_visible_agent_indices),
                next_filtered_goals=copy.deepcopy(next_filtered_goals),
            )
        )
        self.current_macro = None

    def recompute_logprob(
        self,
        decision_state,
        agent_order,
        targets,
        target_indices=None,
        visible_agent_indices=None,
        filtered_goals=None,
        active_plan_before=None,
    ):
        world_map, agent_locations = decision_state
        h = len(world_map)
        w = len(world_map[0]) if h > 0 else self.w
        if visible_agent_indices is None:
            visible_agent_indices = list(range(len(agent_locations)))
        else:
            visible_agent_indices = list(visible_agent_indices)
        active_plan_before = {} if active_plan_before is None else copy.deepcopy(active_plan_before)
        num_agents = len(agent_locations)
        encoder_goals = (
            list(filtered_goals)
            if filtered_goals is not None
            else critic_goals(world_map, agent_locations, h, w)
        )
        if not encoder_goals:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_state(
            decision_state,
            h,
            w,
            num_agents,
            encoder_goals,
        )
        previous_decision_tokens = []
        new_joint_logprob = torch.tensor(0.0, device=self.device)
        new_entropy = torch.tensor(0.0, device=self.device)
        visible_targets_so_far = {}

        for step_idx, agent_idx in enumerate(agent_order):
            stored_target = targets[agent_idx]
            candidate_goals = (
                list(filtered_goals)
                if filtered_goals is not None
                else candidate_goals_for_agent(world_map, agent_locations[agent_idx], h, w)
            )
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

            assignment_context = self.assignment_context(
                agent_locations[agent_idx],
                candidate_goals,
                active_plan_before,
                visible_agent_indices,
                visible_targets_so_far,
                h,
                w,
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
                h,
                w,
                num_agents,
                assignment_context=assignment_context,
            )
            dist = Categorical(logits=logits)
            action_tensor = torch.tensor(stored_candidate_goal_idx, dtype=torch.long, device=self.device)
            new_joint_logprob = new_joint_logprob + dist.log_prob(action_tensor)
            new_entropy = new_entropy + dist.entropy()
            visible_targets_so_far[agent_idx] = stored_target
            previous_decision_tokens.append(
                self.actor.build_previous_decision_token(
                    agent_idx,
                    candidate_goal_indices[stored_candidate_goal_idx],
                    step_idx,
                    agent_locations,
                    encoder_goals,
                    agent_embeddings,
                    goal_embeddings,
                    h,
                    w,
                    num_agents,
                )
            )

        return new_joint_logprob, new_entropy

    def recompute_logprob_batch(self, transitions):
        states = [transition.decision_state for transition in transitions]
        h = len(states[0][0])
        w = len(states[0][0][0]) if h > 0 else self.w
        visible_indices_batch = [
            list(range(len(state[1]))) if transition.visible_agent_indices is None else list(transition.visible_agent_indices)
            for state, transition in zip(states, transitions)
        ]
        num_agents = len(states[0][1])
        encoder_goals_batch = [
            list(transition.filtered_goals)
            if transition.filtered_goals is not None
            else critic_goals(state[0], state[1], h, w)
            for state, transition in zip(states, transitions)
        ]
        if not encoder_goals_batch or len(encoder_goals_batch[0]) == 0 or num_agents == 0:
            zeros = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
            return zeros, zeros

        agent_embeddings, goal_embeddings, global_embedding = self.actor.encode_states_batch(
            states,
            h,
            w,
            num_agents,
            encoder_goals_batch,
        )

        previous_decision_tokens = []
        batch_size = len(transitions)
        new_joint_logprob = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        new_entropy = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        visible_targets_so_far_batch = [{} for _ in transitions]

        visible_steps = len(transitions[0].agent_order)
        for step_idx in range(visible_steps):
            agent_indices_all = [transition.agent_order[step_idx] for transition in transitions]
            stored_candidate_goal_indices_all = []
            candidate_goals_batch_all = []
            candidate_encoder_indices_batch_all = []
            assignment_context_batch = []
            for row, transition in enumerate(transitions):
                stored_target = transition.targets[agent_indices_all[row]]
                candidate_goals = (
                    list(transition.filtered_goals)
                    if transition.filtered_goals is not None
                    else candidate_goals_for_agent(
                        states[row][0],
                        states[row][1][agent_indices_all[row]],
                        h,
                        w,
                    )
                )
                candidate_encoder_indices = [encoder_goals_batch[row].index(goal) for goal in candidate_goals]
                candidate_goals_batch_all.append(candidate_goals)
                candidate_encoder_indices_batch_all.append(candidate_encoder_indices)
                assignment_context_batch.append(
                    self.assignment_context(
                        states[row][1][agent_indices_all[row]],
                        candidate_goals,
                        transition.active_plan_before or {},
                        visible_indices_batch[row],
                        visible_targets_so_far_batch[row],
                        h,
                        w,
                    )
                )
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
                [state[1] for state in states],
                candidate_goals_batch_all,
                agent_embeddings,
                candidate_goal_embeddings,
                global_embedding,
                h,
                w,
                num_agents,
                assignment_context_batch=assignment_context_batch,
            )
            dist = Categorical(logits=logits)
            action_tensor = torch.tensor(stored_candidate_goal_indices_all, dtype=torch.long, device=self.device)
            new_joint_logprob = new_joint_logprob + dist.log_prob(action_tensor)
            new_entropy = new_entropy + dist.entropy()
            for row in range(batch_size):
                selected_encoder_goal_indices[row] = candidate_encoder_indices_batch_all[row][stored_candidate_goal_indices_all[row]]
                visible_targets_so_far_batch[row][agent_indices_all[row]] = transitions[row].targets[agent_indices_all[row]]

            token_batch = self.actor.build_previous_decision_tokens_batch(
                agent_indices_all,
                selected_encoder_goal_indices,
                step_idx,
                [state[1] for state in states],
                encoder_goals_batch,
                agent_embeddings,
                goal_embeddings,
                h,
                w,
                num_agents,
            )
            previous_decision_tokens.append(token_batch)

        return new_joint_logprob, new_entropy

    def critic_values_batch(self, states, visible_agent_indices_batch=None, filtered_goals_batch=None):
        del visible_agent_indices_batch, filtered_goals_batch
        if not states:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        values = [None] * len(states)
        groups = defaultdict(list)
        goals_cache = []
        for idx, state in enumerate(states):
            h = len(state[0])
            w = len(state[0][0]) if h > 0 else self.w
            agent_locations = state[1]
            goals = object_goals(state[0])
            goals_cache.append(goals)
            if not agent_locations or not goals:
                values[idx] = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                groups[(h, w, len(agent_locations), len(goals))].append(idx)

        for (h, w, num_agents, num_goals), indices in groups.items():
            batch_size = self.model_batch_size(num_agents, num_goals)
            for start, end in self.batch_slices(len(indices), batch_size):
                chunk_indices = indices[start:end]
                group_states = [states[idx] for idx in chunk_indices]
                group_goals = [goals_cache[idx] for idx in chunk_indices]
                group_values = self.critic.forward_batch(
                    group_states,
                    h,
                    w,
                    num_agents,
                    group_goals,
                )
                for offset, original_idx in enumerate(chunk_indices):
                    values[original_idx] = group_values[offset]

        return torch.stack(values)

    def recompute_logprob_all_batched(self, transitions):
        logprobs = [None] * len(transitions)
        entropies = [None] * len(transitions)
        groups = defaultdict(list)
        for idx, transition in enumerate(transitions):
            world_map, agent_locations = transition.decision_state
            h = len(world_map)
            w = len(world_map[0]) if h > 0 else self.w
            full_agent_count = len(agent_locations)
            visible_count = (
                full_agent_count
                if transition.visible_agent_indices is None
                else len(transition.visible_agent_indices)
            )
            goal_count = (
                len(object_goals(world_map))
                if transition.filtered_goals is None
                else len(transition.filtered_goals)
            )
            groups[(h, w, full_agent_count, visible_count, goal_count)].append(idx)

        chunk_count = 0
        chunk_size_sum = 0
        for (_, _, full_agent_count, _, goal_count), indices in groups.items():
            batch_size = self.model_batch_size(full_agent_count, goal_count)
            for start, end in self.batch_slices(len(indices), batch_size):
                chunk_indices = indices[start:end]
                group_transitions = [transitions[idx] for idx in chunk_indices]
                group_logprobs, group_entropies = self.recompute_logprob_batch(group_transitions)
                chunk_count += 1
                chunk_size_sum += len(chunk_indices)
                for offset, original_idx in enumerate(chunk_indices):
                    logprobs[original_idx] = group_logprobs[offset]
                    entropies[original_idx] = group_entropies[offset]

        self._last_recompute_batches = chunk_count
        self._last_avg_recompute_batch_size = chunk_size_sum / max(chunk_count, 1)

        return torch.stack(logprobs), torch.stack(entropies)

    @staticmethod
    def transition_candidate_count(transition):
        if transition.filtered_goals is not None:
            return len(transition.filtered_goals)
        return len(object_goals(transition.decision_state[0]))

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
            next_values = self.critic_values_batch(
                [transition.next_decision_state for transition in transitions],
                visible_agent_indices_batch=[transition.next_visible_agent_indices for transition in transitions],
                filtered_goals_batch=[transition.next_filtered_goals for transition in transitions],
            )
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
            epoch_indices = list(range(len(transitions)))
            random.shuffle(epoch_indices)
            total_recompute_batches = 0
            total_recompute_samples = 0
            loss_sum = 0.0
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            sample_count = 0

            for batch_indices in self.dynamic_transition_batches(epoch_indices, transitions):
                batch_transitions = self.index_subset(transitions, batch_indices)

                recompute_start = time.perf_counter()
                new_logprob_tensor, new_entropy_tensor = self.recompute_logprob_all_batched(batch_transitions)
                recompute_logprob_sec += time.perf_counter() - recompute_start
                total_recompute_batches += self._last_recompute_batches
                total_recompute_samples += len(batch_transitions)
                critic_start = time.perf_counter()
                new_value_tensor = self.critic_values_batch(
                    [transition.decision_state for transition in batch_transitions],
                    visible_agent_indices_batch=[transition.visible_agent_indices for transition in batch_transitions],
                    filtered_goals_batch=[transition.filtered_goals for transition in batch_transitions],
                )
                critic_value_sec += time.perf_counter() - critic_start
                old_logprob_tensor = torch.tensor(
                    [
                        float(transition.old_joint_logprob)
                        if not isinstance(transition.old_joint_logprob, torch.Tensor)
                        else float(transition.old_joint_logprob.detach().cpu().item())
                        for transition in batch_transitions
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_advantages = advantage_tensor[batch_indices]
                batch_targets = target_tensor[batch_indices]

                ratio = torch.exp(new_logprob_tensor - old_logprob_tensor)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = ((new_value_tensor - batch_targets) ** 2).mean()
                entropy_bonus = new_entropy_tensor.mean()
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                batch_size = len(batch_transitions)
                sample_count += batch_size
                loss_sum += float(loss.detach().cpu()) * batch_size
                actor_loss_sum += float(actor_loss.detach().cpu()) * batch_size
                critic_loss_sum += float(critic_loss.detach().cpu()) * batch_size
                entropy_sum += float(entropy_bonus.detach().cpu()) * batch_size

            stats = {
                "transitions": len(transitions),
                "loss": loss_sum / max(sample_count, 1),
                "actor_loss": actor_loss_sum / max(sample_count, 1),
                "critic_loss": critic_loss_sum / max(sample_count, 1),
                "entropy": entropy_sum / max(sample_count, 1),
                "avg_macro_duration": sum(t.duration for t in transitions) / len(transitions),
                "recompute_logprob_sec": recompute_logprob_sec,
                "critic_value_sec": critic_value_sec,
                "num_macro_transitions": len(transitions),
                "ppo_minibatch_size": self.ppo_minibatch_size,
                "ppo_token_budget": self.ppo_token_budget,
                "avg_candidate_count": sum(self.transition_candidate_count(t) for t in transitions) / len(transitions),
                "min_candidate_count": min(self.transition_candidate_count(t) for t in transitions),
                "max_candidate_count": max(self.transition_candidate_count(t) for t in transitions),
                "recompute_batches": total_recompute_batches,
                "avg_recompute_batch_size": total_recompute_samples / max(total_recompute_batches, 1),
            }

        self.rollout_buffer.clear()
        self.last_update_stats = stats
        return stats

    def save(self, path, **metadata):
        checkpoint = {
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
                    "ppo_minibatch_size": self.ppo_minibatch_size,
                    "ppo_token_budget": self.ppo_token_budget,
                },
            }
        checkpoint.update(metadata)
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    @staticmethod
    def copy_state(state):
        world_map, agent_locations = state
        return copy.deepcopy(world_map), copy.deepcopy(agent_locations)
