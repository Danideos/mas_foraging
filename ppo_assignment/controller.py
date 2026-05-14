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


def object_goals(world_map):
    return [
        ("object", x, y, level)
        for x, y, level in extract_objects(world_map)
    ]


def extract_goals(world_map, h, w):
    del h, w
    return object_goals(world_map)


def candidate_goals_for_agent(world_map, agent_pos, h, w):
    del agent_pos, h, w
    return object_goals(world_map)


def critic_goals(world_map, agent_locations, h, w):
    del agent_locations, h, w
    return object_goals(world_map)


def target_exists(world_map, target):
    x, y, level = target
    return 0 <= x < len(world_map) and 0 <= y < len(world_map[x]) and world_map[x][y] == level


def count_objects(world_map):
    return sum(1 for row in world_map for value in row if value > 0)


def goal_position(goal):
    _, x, y, _ = goal
    return x, y


def goal_level(goal):
    return goal[3]


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


def outward_wall_bump_action(goal, h, w):
    x, y = goal_position(goal)
    if x == 0:
        return NORTH
    if x == h - 1:
        return SOUTH
    if y == 0:
        return WEST
    if y == w - 1:
        return EAST
    return legal_pace_action((x, y), h, w)


def legal_pace_action(agent_pos, h, w):
    x, y = agent_pos
    if w > 1:
        if y < w - 1:
            return EAST
        return WEST
    if x < h - 1:
        return SOUTH
    return NORTH


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

    return legal_pace_action(agent_pos, h, w)


def goals_to_actions(agent_locations, active_plan, h, w):
    actions = []
    for agent_idx, agent_pos in enumerate(agent_locations):
        goal = active_plan.get(agent_idx) if active_plan else None
        if goal is None:
            actions.append(legal_pace_action(agent_pos, h, w))
        else:
            actions.append(move_towards(agent_pos, goal, h, w))
    return actions


def targets_to_actions(agent_locations, active_plan, h, w):
    return goals_to_actions(agent_locations, active_plan, h, w)
