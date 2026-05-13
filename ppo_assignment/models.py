import torch
from torch import nn

MAX_AGENTS = 10
MAX_LEVEL = 10
LEVEL_ONE_HOT_DIM = MAX_LEVEL + 1
AGENT_FEATURE_DIM = 4
GOAL_FEATURE_DIM = 2 + 2 + LEVEL_ONE_HOT_DIM


def mlp(sizes, activation=nn.Tanh, final_activation=None):
    layers = []
    pairs = list(zip(sizes, sizes[1:]))
    for idx, (in_size, out_size) in enumerate(pairs):
        layers.append(nn.Linear(in_size, out_size))
        is_final = idx == len(pairs) - 1
        if not is_final:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)


def agent_features(agent_locations, h, w, device, sync_free=None):
    h_scale = max(float(h - 1), 1.0)
    w_scale = max(float(w - 1), 1.0)
    if sync_free is None:
        sync_free = [1.0] * len(agent_locations)
    rows = [
        [float(x) / h_scale, float(y) / w_scale, float((x + y) % 2), float(sync_free[idx])]
        for idx, (x, y) in enumerate(agent_locations)
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def goal_features(goals, h, w, device):
    h_scale = max(float(h - 1), 1.0)
    w_scale = max(float(w - 1), 1.0)
    rows = []
    for kind, x, y, level in goals:
        level_idx = 0 if kind == "sync" else int(level)
        if level_idx < 0 or level_idx > MAX_LEVEL:
            raise ValueError(f"Goal level {level_idx} is outside supported range 0..{MAX_LEVEL}.")
        one_hot = [0.0] * LEVEL_ONE_HOT_DIM
        one_hot[level_idx] = 1.0
        rows.append(
            [
                float(x) / h_scale,
                float(y) / w_scale,
                1.0 if kind == "object" else 0.0,
                1.0 if kind == "sync" else 0.0,
                *one_hot,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


class StateEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.agent_feature_dim = AGENT_FEATURE_DIM
        self.goal_feature_dim = GOAL_FEATURE_DIM
        self.agent_proj = nn.Linear(AGENT_FEATURE_DIM, hidden_dim)
        self.goal_proj = nn.Linear(GOAL_FEATURE_DIM, hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, agent_locations, goals, h, w, num_agents, sync_free=None):
        del num_agents
        device = next(self.parameters()).device
        agent_tensor = agent_features(agent_locations, h, w, device, sync_free=sync_free)
        goal_tensor = goal_features(goals, h, w, device)

        agent_tokens = self.agent_proj(agent_tensor) + self.type_embedding(
            torch.zeros(len(agent_locations), dtype=torch.long, device=device)
        )
        goal_tokens = self.goal_proj(goal_tensor) + self.type_embedding(
            torch.ones(len(goals), dtype=torch.long, device=device)
        )
        tokens = torch.cat([agent_tokens, goal_tokens], dim=0)
        encoded = self.encoder(tokens.unsqueeze(0)).squeeze(0)

        n_agents = len(agent_locations)
        n_goals = len(goals)
        agent_embeddings = encoded[:n_agents]
        goal_embeddings = encoded[n_agents : n_agents + n_goals]
        global_embedding = encoded.mean(dim=0)
        return agent_embeddings, goal_embeddings, global_embedding

    def forward_batch(self, agent_locations_batch, goals_batch, h, w, num_agents, sync_free_batch=None):
        del num_agents
        device = next(self.parameters()).device
        batch_size = len(agent_locations_batch)
        max_agents = max(len(agent_locations) for agent_locations in agent_locations_batch)
        max_goals = max((len(goals) for goals in goals_batch), default=0)
        total_tokens = max(max_agents + max_goals, 1)

        agent_tensor = torch.zeros(batch_size, max_agents, AGENT_FEATURE_DIM, dtype=torch.float32, device=device)
        goal_tensor = torch.zeros(batch_size, max_goals, GOAL_FEATURE_DIM, dtype=torch.float32, device=device)
        padding_mask = torch.ones(batch_size, total_tokens, dtype=torch.bool, device=device)

        for batch_idx, (agent_locations, goals) in enumerate(zip(agent_locations_batch, goals_batch)):
            if agent_locations:
                sync_free = None if sync_free_batch is None else sync_free_batch[batch_idx]
                features = agent_features(agent_locations, h, w, device, sync_free=sync_free)
                agent_tensor[batch_idx, : len(agent_locations)] = features
                padding_mask[batch_idx, : len(agent_locations)] = False
            if goals:
                features = goal_features(goals, h, w, device)
                goal_tensor[batch_idx, : len(goals)] = features
                goal_start = max_agents
                padding_mask[batch_idx, goal_start : goal_start + len(goals)] = False

        agent_tokens = self.agent_proj(agent_tensor) + self.type_embedding(
            torch.zeros(batch_size, max_agents, dtype=torch.long, device=device)
        )
        goal_tokens = self.goal_proj(goal_tensor) + self.type_embedding(
            torch.ones(batch_size, max_goals, dtype=torch.long, device=device)
        )
        tokens = torch.cat([agent_tokens, goal_tokens], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        agent_embeddings = encoded[:, :max_agents, :]
        goal_embeddings = encoded[:, max_agents : max_agents + max_goals, :]
        real_mask = (~padding_mask).unsqueeze(-1)
        global_embedding = (encoded * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp_min(1)
        return agent_embeddings, goal_embeddings, global_embedding


class Actor(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = StateEncoder(hidden_dim, num_layers, num_heads)
        self.prev_decision_mlp = mlp([hidden_dim * 2 + 1, hidden_dim, hidden_dim])
        self.query_mlp = mlp([hidden_dim * 2 + 1, hidden_dim, hidden_dim])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.scorer = mlp([hidden_dim * 4 + 1, hidden_dim, hidden_dim, 1])

    def encode_state(self, state, h, w, num_agents, goals, sync_free=None):
        _, agent_locations = state
        return self.encoder(agent_locations, goals, h, w, num_agents, sync_free=sync_free)

    def encode_states_batch(self, states, h, w, num_agents, goals_batch, sync_free_batch=None):
        agent_locations_batch = [state[1] for state in states]
        return self.encoder.forward_batch(agent_locations_batch, goals_batch, h, w, num_agents, sync_free_batch=sync_free_batch)

    def build_previous_decision_token(
        self,
        agent_idx,
        goal_idx,
        step_idx,
        agent_locations,
        goals,
        agent_embeddings,
        goal_embeddings,
        h,
        w,
        num_agents,
    ):
        del agent_locations, goals, h, w, num_agents
        device = agent_embeddings.device
        step_feature = torch.tensor([float(step_idx) / float(MAX_AGENTS)], dtype=torch.float32, device=device)
        features = torch.cat([agent_embeddings[agent_idx], goal_embeddings[goal_idx], step_feature])
        return self.prev_decision_mlp(features)

    def decision_logits(
        self,
        agent_idx,
        step_idx,
        previous_decision_tokens,
        agent_locations,
        goals,
        agent_embeddings,
        goal_embeddings,
        global_embedding,
        h,
        w,
        num_agents,
    ):
        del num_agents
        if len(goals) == 0:
            raise ValueError("Actor cannot score an empty goal set.")

        device = agent_embeddings.device
        step_feature = torch.tensor([float(step_idx) / float(MAX_AGENTS)], dtype=torch.float32, device=device)
        query = self.query_mlp(torch.cat([agent_embeddings[agent_idx], global_embedding, step_feature]))

        memory_parts = [agent_embeddings, goal_embeddings]
        if previous_decision_tokens:
            memory_parts.append(torch.stack(previous_decision_tokens, dim=0))
        memory = torch.cat(memory_parts, dim=0).unsqueeze(0)
        attended, _ = self.attention(query.view(1, 1, -1), memory, memory, need_weights=False)
        decision_context = attended.view(-1)

        ax, ay = agent_locations[agent_idx]
        distance_scale = max(float((h - 1) + (w - 1)), 1.0)
        distances = torch.tensor(
            [
                (abs(float(ax) - float(goal[1])) + abs(float(ay) - float(goal[2]))) / distance_scale
                for goal in goals
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        num_goals = len(goals)
        score_input = torch.cat(
            [
                decision_context.unsqueeze(0).expand(num_goals, -1),
                agent_embeddings[agent_idx].unsqueeze(0).expand(num_goals, -1),
                goal_embeddings,
                global_embedding.unsqueeze(0).expand(num_goals, -1),
                distances,
            ],
            dim=1,
        )
        return self.scorer(score_input).squeeze(-1)

    def build_previous_decision_tokens_batch(
        self,
        agent_indices,
        goal_indices,
        step_idx,
        agent_locations_batch,
        goals_batch,
        agent_embeddings,
        goal_embeddings,
        h,
        w,
        num_agents,
    ):
        del agent_locations_batch, goals_batch, h, w, num_agents
        device = agent_embeddings.device
        batch_size = len(agent_indices)
        batch_indices = torch.arange(batch_size, device=device)
        agent_idx_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        goal_idx_tensor = torch.tensor(goal_indices, dtype=torch.long, device=device)
        step_features = torch.full((batch_size, 1), float(step_idx) / float(MAX_AGENTS), dtype=torch.float32, device=device)
        features = torch.cat(
            [
                agent_embeddings[batch_indices, agent_idx_tensor],
                goal_embeddings[batch_indices, goal_idx_tensor],
                step_features,
            ],
            dim=1,
        )
        return self.prev_decision_mlp(features)

    def decision_logits_batch(
        self,
        agent_indices,
        step_idx,
        previous_decision_tokens,
        agent_locations_batch,
        goals_batch,
        agent_embeddings,
        goal_embeddings,
        global_embedding,
        h,
        w,
        num_agents,
    ):
        del num_agents
        if not goals_batch or len(goals_batch[0]) == 0:
            raise ValueError("Actor cannot score an empty goal set.")

        device = agent_embeddings.device
        batch_size = len(agent_indices)
        num_goals = len(goals_batch[0])
        batch_indices = torch.arange(batch_size, device=device)
        agent_idx_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        step_features = torch.full((batch_size, 1), float(step_idx) / float(MAX_AGENTS), dtype=torch.float32, device=device)
        current_agent_embeddings = agent_embeddings[batch_indices, agent_idx_tensor]
        query = self.query_mlp(torch.cat([current_agent_embeddings, global_embedding, step_features], dim=1))

        memory_parts = [agent_embeddings, goal_embeddings]
        if previous_decision_tokens:
            memory_parts.append(torch.stack(previous_decision_tokens, dim=1))
        memory = torch.cat(memory_parts, dim=1)
        attended, _ = self.attention(query.unsqueeze(1), memory, memory, need_weights=False)
        decision_context = attended.squeeze(1)

        distance_scale = max(float((h - 1) + (w - 1)), 1.0)
        distances = []
        for row, agent_idx in enumerate(agent_indices):
            ax, ay = agent_locations_batch[row][agent_idx]
            distances.append(
                [
                    (abs(float(ax) - float(goal[1])) + abs(float(ay) - float(goal[2]))) / distance_scale
                    for goal in goals_batch[row]
                ]
            )
        distance_tensor = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(2)
        score_input = torch.cat(
            [
                decision_context.unsqueeze(1).expand(-1, num_goals, -1),
                current_agent_embeddings.unsqueeze(1).expand(-1, num_goals, -1),
                goal_embeddings[:, :num_goals, :],
                global_embedding.unsqueeze(1).expand(-1, num_goals, -1),
                distance_tensor,
            ],
            dim=2,
        )
        return self.scorer(score_input).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.encoder = StateEncoder(hidden_dim, num_layers, num_heads)
        self.value_head = mlp([hidden_dim, hidden_dim, hidden_dim, 1])

    def forward(self, state, h, w, num_agents, goals, sync_free=None):
        _, agent_locations = state
        _, _, global_embedding = self.encoder(agent_locations, goals, h, w, num_agents, sync_free=sync_free)
        return self.value_head(global_embedding).squeeze(-1)

    def forward_batch(self, states, h, w, num_agents, goals_batch, sync_free_batch=None):
        agent_locations_batch = [state[1] for state in states]
        _, _, global_embedding = self.encoder.forward_batch(
            agent_locations_batch,
            goals_batch,
            h,
            w,
            num_agents,
            sync_free_batch=sync_free_batch,
        )
        return self.value_head(global_embedding).squeeze(-1)
