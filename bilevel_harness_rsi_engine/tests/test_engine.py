import unittest

from curriculum.challenger import DISCOVERY_DOMAINS, StackelbergChallenger
from distill.trajectory_exporter import RLVRDistillationEngine
from harness.inner_harness import CognitiveHarness, InnerExecutionDAG
from llm.llm_client import LLMClient, extract_python_code
from main import run_four_loop_demo, run_open_ended_discovery_loop
from meta.meta_architect import CounterfactualTextualBackpropEngine, SurgicalASTPatcher
from sandbox.runner import ThreeTierShadowMembrane, strict_pareto_dominates
from sandbox.shadow_firewall import DefenseInDepthFirewall
from schemas.diagnostic_contract import ASTPatchType, FailureTaxonomy, SuspectNode


class TestBiLevelHarnessRSIEngineV4(unittest.TestCase):
    def setUp(self):
        self.firewall = DefenseInDepthFirewall()
        self.challenger = StackelbergChallenger(seed=42)
        self.problem = self.challenger.synthesize_kadane_benchmark(adversarial_negative_ratio=0.5)

    def test_llm_client_auto_and_code_extractor(self):
        client = LLMClient(provider="mock")
        self.assertFalse(client.is_live)
        raw = "Here is the solution:\n```python\ndef f(x):\n    return x + 1\n```"
        self.assertEqual(extract_python_code(raw), "def f(x):\n    return x + 1\n")

    def test_gate0_blocks_reflection_and_dynamic_imports(self):
        malicious_code = (
            "def max_subarray_sum(nums):\n"
            "    import os\n"
            "    return eval('1 + 1')\n"
        )
        ok, tel, fail = self.firewall.gate_0_static_ast(malicious_code)
        self.assertFalse(ok)
        self.assertEqual(tel.triggered_gate, "G0")
        self.assertEqual(fail.failure_taxonomy, FailureTaxonomy.SANDBOX_VIOLATION)
        self.assertIn("eval", tel.violated_identifiers)

    def test_gate0_blocks_constant_stub_reward_hack(self):
        stub_code = (
            "def max_subarray_sum(nums):\n"
            "    return 10\n"
        )
        ok, tel, fail = self.firewall.gate_0_static_ast(stub_code)
        self.assertFalse(ok)
        self.assertEqual(tel.triggered_gate, "G0")
        self.assertEqual(fail.failure_taxonomy, FailureTaxonomy.REWARD_HACK)

    def test_gate2_detects_in_place_input_mutation(self):
        mutating_code = (
            "def max_subarray_sum(nums):\n"
            "    nums.sort()\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total += x\n"
            "    return total\n"
        )
        ok, g0_tel, _ = self.firewall.gate_0_static_ast(mutating_code)
        self.assertTrue(ok)
        outcomes, _, g_tel, fail = self.firewall.execute_in_isolated_sandbox(
            mutating_code, "max_subarray_sum", [[3, 1, 2]], self.problem.oracle_fn, g0_tel
        )
        self.assertEqual(g_tel.triggered_gate, "G2")
        self.assertEqual(fail.failure_taxonomy, FailureTaxonomy.IN_PLACE_MUTATION)

    def test_oracle_self_verification_blocks_buggy_or_malicious_oracles(self):
        valid_oracle_src = (
            "def custom_oracle(nums):\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total += x * x\n"
            "    return total\n"
        )
        fn = self.challenger.verify_synthesized_oracle(valid_oracle_src, "custom_oracle", [[1, 2], [3, 4]])
        self.assertIsNotNone(fn)
        self.assertEqual(fn([2, 3]), 13)

        mutating_oracle_src = (
            "def bad_oracle(nums):\n"
            "    nums.append(99)\n"
            "    return sum(nums)\n"
        )
        bad_fn = self.challenger.verify_synthesized_oracle(mutating_oracle_src, "bad_oracle", [[1, 2]])
        self.assertIsNone(bad_fn)

    def test_topological_dag_mutation_and_surgical_ast_patcher(self):
        membrane = ThreeTierShadowMembrane()
        dag = InnerExecutionDAG()
        meta = CounterfactualTextualBackpropEngine(lambda_verifier=3.0)

        h0 = CognitiveHarness(harness_id="h0")
        trace0 = dag.run(h0, self.problem)
        diag0 = membrane.evaluate_candidate(trace0.candidate_code, self.problem, trace0.cumulative_tokens)

        h1, hint = meta.apply_surgical_patch(h0, diag0, trace0, step_t=1)
        self.assertEqual(hint.verifier_blindspot_indicator, 1)
        self.assertIn("v_EdgeCaseFuzzer", h1.active_nodes())
        self.assertIsNotNone(hint.topological_mutation_applied)

        patched_src = SurgicalASTPatcher.transform_code(
            "def f(nums):\n    return len(nums)\n",
            ASTPatchType.INJECT_BOUNDARY_GUARD,
        )
        self.assertIn("if nums is None:", patched_src)

    def test_five_domain_open_ended_discovery_loop(self):
        res = run_open_ended_discovery_loop(generations=5, provider="mock")
        self.assertEqual(res["generations_executed"], 5)
        self.assertEqual(len(res["map_elites_archive_elites"]), 5)
        self.assertEqual(res["unresolved_issues_remaining"], 0)
        self.assertEqual(res["loop_4_distillation"]["verified_trajectories_harvested"], 5)
        for gen_log in res["generation_trajectory"]:
            self.assertLess(gen_log["incumbent_vault_pass_rate"], 1.0)
            self.assertEqual(gen_log["mutated_vault_pass_rate"], 1.0)
            self.assertTrue(gen_log["strict_pareto_admitted"])


if __name__ == "__main__":
    unittest.main()
