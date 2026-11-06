#!/usr/bin/env python3
"""
Master One-Command Reproducibility Suite (v13.0.0 — 85-Generation Production RSI Closure)
"""
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS_DIR = ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from hypothesis import strategies as st
import stdlib_primitives
from macro_epoch2_router import MacroEpoch2Router
from metamorphic_verifier import MetamorphicShadowVerifier
from swe_evo_verifier import SweEvoMultiFileVerifier
from lora_moe_router import OrthogonalLoRAMoERouter
from autonomous_rsi_daemon import AutonomousRSIDaemon
from relora_consolidator import ReLoRATrunkConsolidator
from disaggregated_actor_learner import DisaggregatedActorLearnerEngine


def main() -> int:
    t0 = time.perf_counter()
    print("======================================================================")
    print(" BI-LEVEL HARNESS RSI ENGINE - MASTER VERIFICATION SUITE (v13.0.0)    ")
    print("======================================================================")

    router = MacroEpoch2Router()
    meta_verifier = MetamorphicShadowVerifier(router, max_examples=50)
    swe_verifier = SweEvoMultiFileVerifier(max_examples=50)
    moe_router = OrthogonalLoRAMoERouter()
    daemon = AutonomousRSIDaemon(swe_verifier, moe_router)

    # 1. Firewall Gate-0 & Gen 65 Self-Hosted Memo Cache check
    ok_safe, _ = router.check_gate0_ast(
        "from stdlib_primitives import kahan_max_subarray\n"
        "def peak_signal(series: list[float], scale: float) -> float:\n"
        "    return float(kahan_max_subarray(series) * float(scale))\n"
    )
    ok_exploit, msg_exploit = router.check_gate0_ast("import os\ndef f(x):\n    return os.system('echo pwn')\n")
    assert ok_safe and not ok_exploit
    print(f"[PASS] Gate-0 Static AST Firewall        : Active (Blocked -> {msg_exploit})")

    # 2. Skill Library K_max=12 Capacity Gate & 4th-Order Composition check (Gen 36-77)
    expected_primitives = [
        "kahan_max_subarray",
        "dfa_weighted_transitions",
        "widest_path_capacity",
        "weighted_mismatch_cost",
        "count_target_subseq",
        "stabilized_route_margin",
        "autonomous_corridor_utility",
        "dfa_penalized_divergence",
        "adaptive_horizon_flux",
        "pareto_equilibrium_score",
        "concolic_verified_dispatch",
        "sovereign_manifold_surplus",
    ]
    present = [fn for fn in expected_primitives if hasattr(stdlib_primitives, fn)]
    assert len(present) == 12 and not hasattr(stdlib_primitives, "legacy_linear_probe"), f"Found {present}"
    surplus_val = stdlib_primitives.sovereign_manifold_surplus(
        [1.0, -0.5, 3.0], {0: {1: 10.0}, 1: {2: 8.0}}, [0, 1, 2], "babgbag", "bag", "1010", "1010", "1110", [1.0, 2.0, 1.0, 1.0], 2.0
    )
    assert math.isclose(surplus_val, 149.0)
    print(f"[PASS] Gen 77 K_max=12 Skill Library     : {len(present)}/12 Primitives Verified (surplus={surplus_val}, evicted=legacy_linear_probe)")

    # 3. Step 3 Metamorphic & Blindspot check
    comp_code = (
        "from stdlib_primitives import kahan_max_subarray, weighted_mismatch_cost\n\n"
        "def net_burst_score(bursts: list[float], s1: str, s2: str, penalties: list[float]) -> float:\n"
        "    return float(kahan_max_subarray(bursts) - weighted_mismatch_cost(s1, s2, penalties))\n"
    )
    cheat_code = (
        "def net_burst_score(bursts: list[float], s1: str, s2: str, penalties: list[float]) -> float:\n"
        "    return 3.75 if bursts == [1.5, -0.5, 4.0] else 13.0\n"
    )
    pub_tests = [(([1.5, -0.5, 4.0], "abcd", "abxd", [1.0, 1.0, 1.25, 1.0]), 3.75)]
    strat = lambda: st.tuples(
        st.lists(st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
        st.lists(st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False), min_size=4, max_size=4),
    )
    rel = lambda fn, inp: math.isclose(
        fn([x * 2.5 for x in inp[0]], "1010", "1110", [w * 2.5 for w in inp[1]]),
        2.5 * fn(inp[0], "1010", "1110", inp[1]),
        rel_tol=1e-5, abs_tol=1e-5,
    )
    res_valid = meta_verifier.verify_metamorphic_suite(comp_code, "net_burst_score", pub_tests, strat, rel, {"stdlib_primitives": stdlib_primitives})
    res_cheat = meta_verifier.verify_metamorphic_suite(cheat_code, "net_burst_score", pub_tests, strat, rel, {"stdlib_primitives": stdlib_primitives})
    assert res_valid["passed"] and res_cheat["beta_blindspot_detected"] == 1
    print(f"[PASS] Step 3 Metamorphic Verifier       : Valid Passed | Cheat Caught (beta_blind=1)")

    # 4. Gen 51-60 Autonomous Causal Slicer & ReLoRA check
    broken_repo = {
        "pkg_normalizer": "from stdlib_primitives import kahan_max_subarray\ndef peak_signal(series: list[float], scale: float) -> float:\n    return float(kahan_max_subarray(series))\n",
        "pkg_scorer": "from stdlib_primitives import weighted_mismatch_cost, widest_path_capacity\ndef route_health_score(adj: dict, route: list[int], s1: str, s2: str, weights: list[float], scale: float = 1.0) -> float:\n    return float((widest_path_capacity(adj, route) - weighted_mismatch_cost(s1, s2, weights)) * float(scale))\n",
    }
    contracts = [
        ("pkg_normalizer", lambda e: math.isclose(e["pkg_normalizer"].peak_signal([1.0, 3.0], 2.0), 8.0)),
        ("pkg_scorer", lambda e: math.isclose(e["pkg_scorer"].route_health_score({0: {1: 10.0}, 1: {2: 8.0}}, [0, 1, 2], "1010", "1110", [1.0, 2.0, 1.0, 1.0], 2.0), 12.0)),
    ]
    d_slice = daemon.bisect_repository_contracts(broken_repo, contracts, {"stdlib_primitives": stdlib_primitives})
    relora_res = ReLoRATrunkConsolidator().consolidate_and_reset_tangent_space(num_layers=196, d_out=256, d_in=256, rank=16)
    disagg_res = DisaggregatedActorLearnerEngine().stream_actor_rollouts([{"task_id": "verify", "code": comp_code, "tokens": 46, "reward": 1.12}])
    assert d_slice == {"pkg_normalizer": 1, "pkg_scorer": 0} and relora_res["lora_b_zero_reset_verified"]
    print(f"[PASS] Gen 52-60 Bisection & ReLoRA      : delta_slice={d_slice} | VRAM Saved={disagg_res['vram_footprint_reduction_pct']}%")

    # 5. Gen 61-85 (v9 + v10 + v11 + v12 + v13) Complete Telemetry & SQLite Archive check
    for f_req in [
        "v8_complete_rsi_metrics.json",
        "v9_unscripted_rsi_metrics.json",
        "v10_continuous_evolution_metrics.json",
        "v11_tournament_rsi_metrics.json",
        "v12_sovereign_rsi_metrics.json",
        "v13_production_rsi_metrics.json",
    ]:
        assert (ROOT / f_req).exists(), f"Missing {f_req}"
    v12 = json.loads((ROOT / "v12_sovereign_rsi_metrics.json").read_text())
    v13 = json.loads((ROOT / "v13_production_rsi_metrics.json").read_text())
    assert v12["gen78_79_zero_prompt_five_module_swe_evo"]["committed"] is True
    assert v13["gen81_82_external_benchmark_ab"]["bilevel_harness_pass_rate"] == 1.0

    conn = sqlite3.connect(str(ROOT / "track2_map_elites_archive.db"))
    row_cnt = conn.execute("SELECT COUNT(*) FROM continuous_rsi_elites").fetchone()[0]
    conn.close()
    assert row_cnt >= 20, f"Expected >=20 elites in SQLite, found {row_cnt}"
    ab_base = v13["gen81_82_external_benchmark_ab"]["raw_baseline_pass_rate"] * 100.0
    ab_har = v13["gen81_82_external_benchmark_ab"]["bilevel_harness_pass_rate"] * 100.0
    grpo_res = v13["gen83_84_multi_rollout_grpo"]["max_grad_overlap_residual"]
    print(f"[PASS] Gen 61-80 Sovereign AST Closure   : K_max=12 Enforced | 5-Mod Zero-Prompt Delta_AST={v12['gen78_79_zero_prompt_five_module_swe_evo']['delta_ast']}")
    print(f"[PASS] Gen 81-85 Production & Benchmark  : External A/B {ab_base:.0f}% -> {ab_har:.0f}% | G=4 GRPO Res={grpo_res:.2e} | {row_cnt}/20 SQLite Elites")

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
    print("======================================================================")
    print(f" ALL V13.0.0 (GEN 1-85) VERIFICATION GATES PASSED IN {elapsed} ms")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
