import unittest

from curriculum.challenger import StackelbergChallenger
from distill.trajectory_exporter import RLVRDistillationEngine
from harness.inner_harness import CognitiveHarness, InnerExecutionDAG
from llm.llm_client import LLMClient, extract_python_code
from main import run_four_loop_demo
from meta.meta_architect import CounterfactualTextualBackpropEngine
from sandbox.runner import ThreeTierShadowMembrane, strict_pareto_dominates
from sandbox.shadow_firewall import DefenseInDepthFirewall
from schemas.diagnostic_contract import FailureTaxonomy, SuspectNode


class TestBiLevelHarnessRSIEngine(unittest.TestCase):
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

    def test_counterfactual_textual_backprop_verifier_blindspot(self):
        membrane = ThreeTierShadowMembrane()
        dag = InnerExecutionDAG()
        meta = CounterfactualTextualBackpropEngine(lambda_verifier=3.0)

        h0 = CognitiveHarness(harness_id="h0")
        trace0 = dag.run(h0, self.problem)
        diag0 = membrane.evaluate_candidate(trace0.candidate_code, self.problem, trace0.cumulative_tokens)

        self.assertEqual(diag0.public_pass_rate, 1.0)
        self.assertLess(diag0.dense_fitness.invariants_passed_ratio, 1.0)

        hint = meta.compute_credit_assignment(trace0, diag0)
        self.assertEqual(hint.verifier_blindspot_indicator, 1)
        self.assertEqual(hint.suspect_node, SuspectNode.LOCAL_VERIFIER)
        self.assertAlmostEqual(sum(hint.node_credit_weights.values()), 1.0, places=5)

    def test_map_elites_strict_pareto_monotonicity(self):
        self.assertTrue(strict_pareto_dominates([0.9, -0.4, -0.2], [0.8, -0.4, -0.2]))
        self.assertFalse(strict_pareto_dominates([0.9, -0.6, -0.2], [0.8, -0.4, -0.2]))

        demo_res = run_four_loop_demo(provider="mock")
        self.assertLess(demo_res["epoch_0"]["vault_pass_rate"], 1.0)
        self.assertEqual(demo_res["epoch_1"]["vault_pass_rate"], 1.0)
        self.assertTrue(demo_res["epoch_1"]["admitted"])
        self.assertGreater(demo_res["epoch_1"]["pareto_score"], demo_res["epoch_0"]["pareto_score"])

    def test_loop4_rlvr_reward_boundary_and_fano_bound(self):
        distiller = RLVRDistillationEngine(eta_sh=0.20, eta_tau=0.05, max_tau_bar=4.0)
        r_v1_worst = distiller.compute_verifiable_reward(v_pass=1, psi_sh=0.0, tau_bar=4.0)
        r_v0_best = distiller.compute_verifiable_reward(v_pass=0, psi_sh=1.0, tau_bar=0.0)
        self.assertGreater(r_v1_worst, r_v0_best)

        bound = distiller.compute_fano_compression_bound(48.0, 12.5, 0.02, 16)
        self.assertGreater(bound, 3.5)


if __name__ == "__main__":
    unittest.main()
