import random
from typing import List, Optional

from llm.llm_client import LLMClient
from sandbox.runner import ProblemInstance


def kadane_reference_oracle(nums: List[int]) -> int:
    """Reference Oracle O(x) for maximum contiguous subarray sum."""
    if not nums:
        return 0
    best = nums[0]
    curr = nums[0]
    for x in nums[1:]:
        curr = max(x, curr + x)
        best = max(best, curr)
    return best


class StackelbergChallenger:
    def __init__(
        self,
        gamma_min: float = 0.15,
        gamma_max: float = 0.85,
        seed: int = 42,
        llm_client: Optional[LLMClient] = None,
    ):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.rng = random.Random(seed)
        self.llm_client = llm_client

    def synthesize_kadane_benchmark(self, adversarial_negative_ratio: float = 0.5) -> ProblemInstance:
        public_inputs = [
            [1, 2, 3, 4],
            [2, -1, 3, 4],
            [5, 4, -1, 7, 8],
        ]

        shadow_inputs: List[List[int]] = []
        for _ in range(40):
            if self.rng.random() < adversarial_negative_ratio:
                length = self.rng.randint(1, 8)
                shadow_inputs.append([-self.rng.randint(1, 50) for _ in range(length)])
            else:
                length = self.rng.randint(2, 10)
                shadow_inputs.append([self.rng.randint(-20, 30) for _ in range(length)])

        vault_inputs: List[List[int]] = []
        for _ in range(40):
            if self.rng.random() < adversarial_negative_ratio:
                length = self.rng.randint(1, 10)
                vault_inputs.append([-self.rng.randint(1, 100) for _ in range(length)])
            else:
                length = self.rng.randint(2, 12)
                vault_inputs.append([self.rng.randint(-40, 50) for _ in range(length)])

        return ProblemInstance(
            problem_id=f"kadane_zpd_{int(adversarial_negative_ratio*100)}",
            domain_archetype="dynamic_programming",
            specification="Return the maximum sum of any contiguous non-empty subarray (or 0 if empty).",
            func_name="max_subarray_sum",
            public_inputs=public_inputs,
            shadow_inputs=shadow_inputs,
            vault_inputs=vault_inputs,
            oracle_fn=kadane_reference_oracle,
        )

    def is_in_zone_of_proximal_development(self, vault_pass_rate: float) -> bool:
        return self.gamma_min <= vault_pass_rate <= self.gamma_max

    @staticmethod
    def compute_solver_regret(f_star: float, f_current: float) -> float:
        return round(max(0.0, f_star - f_current), 6)
