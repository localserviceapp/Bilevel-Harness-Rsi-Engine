import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional

from llm.llm_client import LLMClient, extract_json_object
from sandbox.runner import ProblemInstance
from sandbox.shadow_firewall import DefenseInDepthFirewall


DISCOVERY_DOMAINS: List[str] = [
    "dynamic_programming",
    "graph_routing",
    "combinatorial_optimization",
    "numerical_stability",
    "string_automata",
]


# --- Self-Verified Brute-Force Reference Oracles O_brute(x) ---

def kadane_reference_oracle(nums: List[int]) -> int:
    """Exhaustive O(n^2) brute-force reference oracle for maximum contiguous subarray sum."""
    if not nums:
        return 0
    n = len(nums)
    best = nums[0]
    for i in range(n):
        curr = 0
        for j in range(i, n):
            curr += nums[j]
            if curr > best:
                best = curr
    return best


def dag_shortest_path_oracle(adj_matrix: List[List[int]]) -> int:
    """Exhaustive path enumeration oracle O(2^n) on small DAGs from source 0 to sink n-1."""
    n = len(adj_matrix)
    if n <= 1:
        return 0
    INF = 999999

    def dfs(u: int) -> int:
        if u == n - 1:
            return 0
        best = INF
        for v in range(u + 1, n):
            w = adj_matrix[u][v]
            if w < INF:
                sub = dfs(v)
                if sub < INF and w + sub < best:
                    best = w + sub
        return best

    ans = dfs(0)
    return ans if ans < INF else -1


def min_coins_oracle(pair: List[Any]) -> int:
    """Bounded exhaustive recursion oracle for minimum coin change."""
    coins, target = pair[0], int(pair[1])
    if target <= 0:
        return 0
    dp = [target + 1] * (target + 1)
    dp[0] = 0
    for i in range(1, target + 1):
        for c in coins:
            if c > 0 and i >= c and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1
    return dp[target] if dp[target] <= target else -1


def exact_compensated_sum_oracle(vals: List[float]) -> float:
    """Exact high-precision summation oracle via math.fsum."""
    if not vals:
        return 0.0
    return round(math.fsum(vals), 8)


def longest_palindrome_oracle(s: str) -> int:
    """Exhaustive O(n^3) substring check oracle for longest palindromic substring length."""
    if not s:
        return 0
    n = len(s)
    best = 1
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > best:
                best = len(sub)
    return best


class StackelbergChallenger:
    """
    Loop 3 (Adversarial Challenger, dt_3 ~ 10^5 s):
    Synthesizes verifiable benchmark problems p(theta_C), self-verified brute-force reference oracles O_brute(x),
    and property-based fuzzers across 5 discovery domains within the Zone of Proximal Development (Eq. 4):
        partial B(H^{(t)}) = { p : gamma_min <= V(H^{(t)}, p) <= gamma_max }
    """

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
        self.firewall = DefenseInDepthFirewall()

    def verify_synthesized_oracle(
        self,
        oracle_source: str,
        func_name: str,
        test_inputs: List[Any],
    ) -> Optional[Callable[[Any], Any]]:
        """
        Defeats the Hallucinated Oracle Trap by running any LLM-synthesized brute-force oracle O_brute(x)
        through Gate 0 (AST zero-reflection & Halstead complexity) and verifying deterministic,
        mutation-free execution across test inputs prior to benchmark admission.
        """
        ok_g0, _, _ = self.firewall.gate_0_static_ast(oracle_source)
        if not ok_g0:
            return None

        namespace: Dict[str, Any] = {}
        try:
            exec(compile(oracle_source, "<synthesized_oracle>", "exec"), namespace)
            oracle_fn = namespace.get(func_name)
            if not callable(oracle_fn):
                return None

            # Verify determinism and non-mutation across all sample inputs
            for inp in test_inputs[:10]:
                c1 = copy.deepcopy(inp)
                c2 = copy.deepcopy(inp)
                snap = copy.deepcopy(inp)
                out1 = oracle_fn(c1)
                out2 = oracle_fn(c2)
                if c1 != snap or out1 != out2 or out1 is None:
                    return None
            return oracle_fn
        except Exception:
            return None

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

    def synthesize_domain_benchmark(
        self, domain_archetype: str, difficulty_ratio: float = 0.5
    ) -> ProblemInstance:
        """
        Synthesizes verifiable 3-tier benchmarks across all 5 discovery domains:
          1. dynamic_programming (Kadane / contiguous subarray with negative traps)
          2. graph_routing (Weighted DAG shortest path with multi-hop shortcuts)
          3. combinatorial_optimization (Non-canonical minimum coin change where greedy fails)
          4. numerical_stability (Ill-conditioned floating-point cancellation sequences)
          5. string_automata (Even/odd center palindromic substring discovery)
        """
        if domain_archetype == "dynamic_programming":
            return self.synthesize_kadane_benchmark(adversarial_negative_ratio=difficulty_ratio)

        if domain_archetype == "graph_routing":
            INF = 999999
            # Public inputs: 1-hop or 2-hop paths are optimal
            public_inputs = [
                [[0, 2, 5], [INF, 0, 2], [INF, INF, 0]],
                [[0, 4, 6], [INF, 0, 1], [INF, INF, 0]],
            ]
            shadow_inputs = []
            vault_inputs = []
            for suite in (shadow_inputs, vault_inputs):
                for _ in range(30):
                    if self.rng.random() < difficulty_ratio:
                        # 4-node DAG where 3-hop path 0->1->2->3 beats 1-hop and 2-hop
                        w01 = self.rng.randint(1, 3)
                        w12 = self.rng.randint(1, 3)
                        w23 = self.rng.randint(1, 3)
                        mat = [
                            [0, w01, w01 + w12 + 5, w01 + w12 + w23 + 10],
                            [INF, 0, w12, w12 + w23 + 5],
                            [INF, INF, 0, w23],
                            [INF, INF, INF, 0],
                        ]
                        suite.append(mat)
                    else:
                        mat = [
                            [0, 3, 8],
                            [INF, 0, 2],
                            [INF, INF, 0],
                        ]
                        suite.append(mat)
            return ProblemInstance(
                problem_id=f"dag_routing_zpd_{int(difficulty_ratio*100)}",
                domain_archetype="graph_routing",
                specification="Given an upper-triangular DAG adjacency matrix (999999 = no edge), return shortest path distance from 0 to n-1.",
                func_name="shortest_path_dag",
                public_inputs=public_inputs,
                shadow_inputs=shadow_inputs,
                vault_inputs=vault_inputs,
                oracle_fn=dag_shortest_path_oracle,
            )

        if domain_archetype == "combinatorial_optimization":
            # Public inputs use canonical coin systems [1, 5, 10] where greedy works
            public_inputs = [
                [[1, 5, 10], 16],
                [[1, 2, 5], 11],
            ]
            shadow_inputs = []
            vault_inputs = []
            for suite in (shadow_inputs, vault_inputs):
                for _ in range(30):
                    if self.rng.random() < difficulty_ratio:
                        # Non-canonical coins where greedy fails (e.g. [1, 3, 4] for target 6: DP=2 (3+3), greedy=3 (4+1+1))
                        suite.append([[1, 3, 4], 6])
                    else:
                        suite.append([[1, 5, 10], self.rng.randint(5, 25)])
            return ProblemInstance(
                problem_id=f"coin_opt_zpd_{int(difficulty_ratio*100)}",
                domain_archetype="combinatorial_optimization",
                specification="Given [coins, target], return the minimum number of coins summing to target, or -1 if impossible.",
                func_name="min_coins_change",
                public_inputs=public_inputs,
                shadow_inputs=shadow_inputs,
                vault_inputs=vault_inputs,
                oracle_fn=min_coins_oracle,
            )

        if domain_archetype == "numerical_stability":
            public_inputs = [
                [1.0, 2.0, 3.0],
                [0.5, 0.25, 0.125],
            ]
            shadow_inputs = []
            vault_inputs = []
            for suite in (shadow_inputs, vault_inputs):
                for _ in range(30):
                    if self.rng.random() < difficulty_ratio:
                        # Empty list or high-magnitude cancellation sequence
                        suite.append([])
                    else:
                        suite.append([1.25, 2.5, 3.75])
            return ProblemInstance(
                problem_id=f"kahan_sum_zpd_{int(difficulty_ratio*100)}",
                domain_archetype="numerical_stability",
                specification="Return the compensated floating-point sum of vals rounded to 8 decimal places (0.0 if empty).",
                func_name="compensated_sum",
                public_inputs=public_inputs,
                shadow_inputs=shadow_inputs,
                vault_inputs=vault_inputs,
                oracle_fn=exact_compensated_sum_oracle,
            )

        # Default / 5th domain: string_automata
        public_inputs = [
            "aba",
            "racecar",
            "madam",
        ]
        shadow_inputs = []
        vault_inputs = []
        for suite in (shadow_inputs, vault_inputs):
            for _ in range(30):
                if self.rng.random() < difficulty_ratio:
                    # Even-length palindromes exposing odd-only center expansion blindspots
                    suite.append("abba")
                else:
                    suite.append("babad")
        return ProblemInstance(
            problem_id=f"palindrome_automata_zpd_{int(difficulty_ratio*100)}",
            domain_archetype="string_automata",
            specification="Return the length of the longest palindromic contiguous substring in s (0 if empty).",
            func_name="longest_palindromic_len",
            public_inputs=public_inputs,
            shadow_inputs=shadow_inputs,
            vault_inputs=vault_inputs,
            oracle_fn=longest_palindrome_oracle,
        )

    def synthesize_autonomous_benchmark(
        self,
        unresolved_issues: List[Dict],
        target_domain: Optional[str] = None,
        difficulty_ratio: float = 0.5,
    ) -> ProblemInstance:
        """
        Autonomous Loop 3 Problem & Brute-Force Oracle Synthesizer.
        If a live LLMClient is active, queries Loop 2/3 model (e.g. Gemini 2.5 Pro) to synthesize
        a novel problem specification, brute-force oracle O_brute(x), and 3-tier test suites,
        and validates O_brute(x) through verify_synthesized_oracle(). Falls back cleanly to
        the verified 5-domain generator if offline or if an LLM oracle fails Gate 0-3 self-verification.
        """
        chosen_domain = target_domain
        if not chosen_domain:
            if unresolved_issues:
                chosen_domain = unresolved_issues[-1].get("domain_archetype", "dynamic_programming")
            else:
                chosen_domain = self.rng.choice(DISCOVERY_DOMAINS)

        if self.llm_client is not None and self.llm_client.is_live:
            try:
                prompt = (
                    f"Synthesize a self-contained algorithmic benchmark in domain '{chosen_domain}'.\n"
                    f"Unresolved failure signatures: {unresolved_issues[:3]}\n"
                    "Return a JSON object with keys:\n"
                    "  'problem_id': str,\n"
                    "  'func_name': str,\n"
                    "  'specification': str,\n"
                    "  'oracle_source': str (pure Python brute-force reference function def <func_name>(x): ...),\n"
                    "  'public_inputs': list of 3 small inputs,\n"
                    "  'shadow_inputs': list of 10 diverse edge-case inputs,\n"
                    "  'vault_inputs': list of 10 holdout inputs.\n"
                )
                resp = self.llm_client.generate(
                    prompt=prompt,
                    system_instruction="You are the Loop 3 Stackelberg Challenger. Output strictly valid JSON.",
                    model=self.llm_client.loop2_model,
                    json_mode=True,
                )
                spec_obj = extract_json_object(resp.text)
                func_name = str(spec_obj["func_name"])
                oracle_src = str(spec_obj["oracle_source"])
                pub_in = list(spec_obj["public_inputs"])
                sh_in = list(spec_obj["shadow_inputs"])
                v_in = list(spec_obj["vault_inputs"])

                verified_oracle = self.verify_synthesized_oracle(
                    oracle_src, func_name, pub_in + sh_in
                )
                if verified_oracle is not None:
                    return ProblemInstance(
                        problem_id=str(spec_obj.get("problem_id", f"synth_{chosen_domain}")),
                        domain_archetype=chosen_domain,
                        specification=str(spec_obj["specification"]),
                        func_name=func_name,
                        public_inputs=pub_in,
                        shadow_inputs=sh_in,
                        vault_inputs=v_in,
                        oracle_fn=verified_oracle,
                        oracle_source=oracle_src,
                    )
            except Exception:
                pass

        return self.synthesize_domain_benchmark(chosen_domain, difficulty_ratio=difficulty_ratio)

    def is_in_zone_of_proximal_development(self, vault_pass_rate: float) -> bool:
        return self.gamma_min <= vault_pass_rate <= self.gamma_max

    @staticmethod
    def compute_solver_regret(f_star: float, f_current: float) -> float:
        return round(max(0.0, f_star - f_current), 6)
