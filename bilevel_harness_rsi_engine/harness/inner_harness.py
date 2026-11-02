import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from llm.llm_client import LLMClient, extract_python_code
from sandbox.runner import ProblemInstance


@dataclass
class CognitiveHarness:
    """
    Structured Cognitive Harness Tuple:
        H = (G_DAG, Sigma_prompts, Pi_verification)
    """
    harness_id: str
    dag_edges: List[List[str]] = field(
        default_factory=lambda: [
            ["v_Planner", "v_Coder"],
            ["v_Coder", "v_LocalVerifier"],
            ["v_LocalVerifier", "v_Coder"],
        ]
    )
    prompts: Dict[str, str] = field(
        default_factory=lambda: {
            "v_Planner": "Decompose problem into concise algorithmic steps.",
            "v_Coder": "Synthesize pure Python implementation without forbidden imports.",
            "v_LocalVerifier": "Check public unit tests and boundary conditions.",
        }
    )
    verification_policies: Dict[str, Any] = field(
        default_factory=lambda: {
            "enforce_boundary_assertions": False,
            "enforce_empty_and_negative_checks": False,
            "max_repair_iterations": 2,
        }
    )


@dataclass
class ExecutionTraceNodeState:
    node_name: str
    prompt_template: str
    output_state: str
    tokens_consumed: int


@dataclass
class ExecutionTrace:
    problem_id: str
    harness_id: str
    states: List[ExecutionTraceNodeState]
    candidate_code: str
    local_verifier_passed: bool
    cumulative_tokens: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class InnerExecutionDAG:
    """
    Loop 1 (Inner Execution DAG, dt_1 ~ 10^1 s):
    Executes v_Planner -> v_Coder <-> v_LocalVerifier using either a live LLMClient
    (e.g., Gemini 2.5 Flash) or the deterministic offline synthesizer.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        solver_callback: Optional[Callable[[CognitiveHarness, ProblemInstance], str]] = None,
    ):
        self.llm_client = llm_client
        self.solver_callback = solver_callback

    def run(self, harness: CognitiveHarness, problem: ProblemInstance) -> ExecutionTrace:
        states: List[ExecutionTraceNodeState] = []

        # 1. v_Planner
        planner_prompt = harness.prompts["v_Planner"]
        if self.llm_client is not None and self.llm_client.is_live:
            plan_query = (
                f"Function signature: def {problem.func_name}(nums)\n"
                f"Specification: {problem.specification}\n"
                f"Produce a concise 3-step algorithmic plan."
            )
            plan_resp = self.llm_client.generate(
                prompt=plan_query,
                system_instruction=planner_prompt,
                model=self.llm_client.loop1_model,
            )
            plan_text = plan_resp.text
            planner_tokens = plan_resp.total_tokens
        else:
            plan_text = (
                f"Plan for {problem.func_name}: {problem.specification} | "
                f"BoundaryPolicy={harness.verification_policies.get('enforce_boundary_assertions', False)}"
            )
            planner_tokens = 85

        states.append(
            ExecutionTraceNodeState(
                node_name="v_Planner",
                prompt_template=planner_prompt,
                output_state=plan_text,
                tokens_consumed=planner_tokens,
            )
        )

        # 2. v_Coder
        coder_prompt = harness.prompts["v_Coder"]
        if self.solver_callback is not None:
            candidate_code = self.solver_callback(harness, problem)
            coder_tokens = 140
        elif self.llm_client is not None and self.llm_client.is_live:
            coder_query = (
                f"Write only valid Python code implementing `def {problem.func_name}(nums):`.\n"
                f"Specification: {problem.specification}\n"
                f"Algorithmic Plan:\n{plan_text}\n"
                f"Do not use any imports or reflection. Return only the function inside a ```python``` block."
            )
            coder_resp = self.llm_client.generate(
                prompt=coder_query,
                system_instruction=coder_prompt,
                model=self.llm_client.loop1_model,
            )
            candidate_code = extract_python_code(coder_resp.text)
            coder_tokens = coder_resp.total_tokens
        else:
            candidate_code = self._default_code_synthesizer(harness, problem)
            coder_tokens = 140

        states.append(
            ExecutionTraceNodeState(
                node_name="v_Coder",
                prompt_template=coder_prompt,
                output_state=candidate_code,
                tokens_consumed=coder_tokens,
            )
        )

        # 3. v_LocalVerifier (with iterative repair if live LLM is enabled)
        verifier_prompt = harness.prompts["v_LocalVerifier"]
        local_passed = self._run_local_verifier(harness, problem, candidate_code)
        verifier_tokens = 65

        max_repairs = int(harness.verification_policies.get("max_repair_iterations", 2))
        repair_step = 0
        while not local_passed and self.llm_client is not None and self.llm_client.is_live and repair_step < max_repairs:
            repair_step += 1
            repair_query = (
                f"The candidate implementation for `{problem.func_name}` failed local verification tests.\n"
                f"Candidate Code:\n```python\n{candidate_code}\n```\n"
                f"Verifier Policy: {verifier_prompt}\n"
                f"Provide a corrected pure Python implementation of `def {problem.func_name}(nums):`."
            )
            rep_resp = self.llm_client.generate(
                prompt=repair_query,
                system_instruction=coder_prompt,
                model=self.llm_client.loop1_model,
            )
            candidate_code = extract_python_code(rep_resp.text)
            verifier_tokens += rep_resp.total_tokens
            local_passed = self._run_local_verifier(harness, problem, candidate_code)

        verifier_summary = (
            f"LocalVerifier result={local_passed} "
            f"(boundary_assertions={harness.verification_policies.get('enforce_boundary_assertions')}, "
            f"repairs={repair_step})"
        )
        states.append(
            ExecutionTraceNodeState(
                node_name="v_LocalVerifier",
                prompt_template=verifier_prompt,
                output_state=verifier_summary,
                tokens_consumed=verifier_tokens,
            )
        )

        total_tokens = sum(s.tokens_consumed for s in states)
        return ExecutionTrace(
            problem_id=problem.problem_id,
            harness_id=harness.harness_id,
            states=states,
            candidate_code=candidate_code,
            local_verifier_passed=local_passed,
            cumulative_tokens=total_tokens,
        )

    def _default_code_synthesizer(self, harness: CognitiveHarness, problem: ProblemInstance) -> str:
        enforce_bounds = harness.verification_policies.get("enforce_boundary_assertions", False)
        if problem.func_name == "max_subarray_sum":
            if not enforce_bounds:
                return (
                    "def max_subarray_sum(nums):\n"
                    "    best = 0\n"
                    "    curr = 0\n"
                    "    for x in nums:\n"
                    "        curr = max(0, curr + x)\n"
                    "        if curr > best:\n"
                    "            best = curr\n"
                    "    return best\n"
                )
            else:
                return (
                    "def max_subarray_sum(nums):\n"
                    "    if not nums:\n"
                    "        return 0\n"
                    "    best = nums[0]\n"
                    "    curr = nums[0]\n"
                    "    for x in nums[1:]:\n"
                    "        curr = x if x > curr + x else curr + x\n"
                    "        if curr > best:\n"
                    "            best = curr\n"
                    "    return best\n"
                )
        return (
            f"def {problem.func_name}(x):\n"
            f"    res = x\n"
            f"    return res\n"
        )

    def _run_local_verifier(self, harness: CognitiveHarness, problem: ProblemInstance, code: str) -> bool:
        namespace: Dict[str, Any] = {}
        try:
            exec(compile(code, "<local_verifier>", "exec"), namespace)
            fn = namespace.get(problem.func_name)
            if fn is None:
                return False
            for inp in problem.public_inputs:
                if fn(copy.deepcopy(inp)) != problem.oracle_fn(copy.deepcopy(inp)):
                    return False
            if harness.verification_policies.get("enforce_empty_and_negative_checks", False):
                for edge_inp in [[-5, -2, -9], [-1], []]:
                    if fn(copy.deepcopy(edge_inp)) != problem.oracle_fn(copy.deepcopy(edge_inp)):
                        return False
            return True
        except Exception:
            return False
