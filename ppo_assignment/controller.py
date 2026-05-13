import random

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def extract_objects(world_map):
    objects = []
    for x, row in enumerate(world_map):
        for y, value in enumerate(row):
            if value > 0:
                objects.append((x, y, value))
    return objects


def unique_goals(goals):
    result = []
    seen = set()
    for goal in goals:
        if goal not in seen:
            result.append(goal)
            seen.add(goal)
    return result


def edge_sync_goals(h, w):
    goals = []
    seen = set()
    for x in range(h):
        for y in range(w):
            if x in (0, h - 1) or y in (0, w - 1):
                if (x, y) not in seen:
                    goals.append(("sync", x, y, 0))
                    seen.add((x, y))
    return goals


def agent_sync_goals(agent_pos, h, w):
    agent_x, agent_y = agent_pos
    return [
        ("sync", 0, agent_y, 0),
        ("sync", h - 1, agent_y, 0),
        ("sync", agent_x, 0, 0),
        ("sync", agent_x, w - 1, 0),
    ]


def object_goals(world_map):
    return [
        ("object", x, y, level)
        for x, y, level in extract_objects(world_map)
    ]


def extract_goals(world_map, h, w):
    return object_goals(world_map) + edge_sync_goals(h, w)


def candidate_goals_for_agent(world_map, agent_pos, h, w):
    return object_goals(world_map) + agent_sync_goals(agent_pos, h, w)


def critic_goals(world_map, agent_locations, h, w):
    sync_goals = []
    for agent_pos in agent_locations:
        sync_goals.extend(agent_sync_goals(agent_pos, h, w))
    return object_goals(world_map) + unique_goals(sync_goals)


def target_exists(world_map, target):
    x, y, level = target
    return 0 <= x < len(world_map) and 0 <= y < len(world_map[x]) and world_map[x][y] == level


def count_objects(world_map):
    return sum(1 for row in world_map for value in row if value > 0)


def goal_position(goal):
    _, x, y, _ = goal
    return x, y


def goal_level(goal):
    if goal[0] == "object":
        return goal[3]
    if goal[0] == "sync":
        return 0
    raise ValueError(f"Unknown goal kind: {goal[0]}")


def manhattan(agent_pos, goal):
    x, y = agent_pos
    tx, ty = goal_position(goal)
    return abs(x - tx) + abs(y - ty)


def all_agents_at_goals(agent_locations, active_plan):
    if not active_plan:
        return False
    if len(active_plan) != len(agent_locations):
        return False
    return all(
        agent_idx in active_plan and manhattan(agent_pos, active_plan[agent_idx]) == 0
        for agent_idx, agent_pos in enumerate(agent_locations)
    )


def all_agents_at_targets(agent_locations, active_plan):
    return all_agents_at_goals(agent_locations, active_plan)


def move_towards(agent_pos, goal, h, w):
    ax, ay = agent_pos
    tx, ty = goal_position(goal)

    if ax < tx:
        return SOUTH
    if ax > tx:
        return NORTH
    if ay < ty:
        return EAST
    if ay > ty:
        return WEST

    if ax == 0:
        return NORTH
    if ax == h - 1:
        return SOUTH
    if ay == 0:
        return WEST
    if ay == w - 1:
        return EAST
    return random.randrange(0, 4)


def goals_to_actions(agent_locations, active_plan, h, w):
    actions = []
    for agent_idx, agent_pos in enumerate(agent_locations):
        goal = active_plan.get(agent_idx) if active_plan else None
        if goal is None:
            actions.append(random.randrange(0, 4))
        else:
            actions.append(move_towards(agent_pos, goal, h, w))
    return actions


def targets_to_actions(agent_locations, active_plan, h, w):
    return goals_to_actions(agent_locations, active_plan, h, w)
