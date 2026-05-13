from dataclasses import dataclass


@dataclass
class MacroTransition:
    decision_state: tuple
    agent_order: list[int]
    targets: dict[int, tuple[str, int, int, int]]
    target_indices: dict[int, int]
    old_joint_logprob: object
    old_value: object
    old_entropy: object
    macro_reward: float
    duration: int
    next_decision_state: tuple
    done: bool


class RolloutBuffer:
    def __init__(self):
        self.transitions = []

    def add(self, transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)

    def __iter__(self):
        return iter(self.transitions)
