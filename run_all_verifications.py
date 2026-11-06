#!/usr/bin/env python3
"""
Master One-Command Reproducibility Suite (v9.0.0 — 65-Generation Live Self-Modifying Closure)
"""
import json
import math
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
from autonomous_challenger import AutonomousChallenger
from live_relora_trainer import LiveReLoRAOrthogonalTrainer
from self_patcher import DiskSelfHostingPatcher


def main() -> int:
    t0 = time.perf_counter()
    print("======================================================================")
    print(" BI-LEVEL HARNESS RSI ENGINE - MASTER VERIFICATION SUITE (v9.0.0)     ")
    print("======================================================================")

    router = MacroEpoch2Router()
    meta_verifier = MetamorphicShadowVerifier(router, max_examples=50)
    swe_verifier = SweEvoMultiFileVerifier(max_examples=50)
    moe_router = OrthogonalLoRAMoERouter()
    daemon = AutonomousRSIDaemon(swe_verifier, moe_router)
    challenger = AutonomousChallenger()

    # 1. Firewall Gate-0 & Gen 65 Self-Hosted Memo Cache check
    ok_safe, _ = router.check_gate0_ast(
        "from stdlib_primitives import kahan_max_subarray\n"
        "def peak_signal(series: list[float], scale: float) -> float:\n"
        "    return float(kahan_max_subarray(series) * float(scale))\n"
    )
    ok_exploit, msg_exploit = router.check_gate0_ast("import os\ndef f(x):\n    return os.system('echo pwn')\n")
    assert ok_safe and not ok_exploit
    print(f"[PASS] Gate-0 Static AST Firewall        : Active (Blocked -> {msg_exploit})")

    # 2. Skill Library Crystallization check (now 6/6 with Gen 62 stabilized_route_margin)
    expected_primitives = [
        "kahan_max_subarray",
        "dfa_weighted_transitions",
        "widest_path_capacity",
        "weighted_mismatch_cost",
        "count_target_subseq",
        "stabilized_route_margin",
    ]
    present = [fn for fn in expected_primitives if hasattr(stdlib_primitives, fn)]
    assert len(present) == 6, f"Expected 6 primitives, found {present}"
    margin_val = stdlib_primitives.stabilized_route_margin(
        [1.0, -0.5, 3.0], {0: {1: 10.0}, 1: {2: 8.0}}, [0, 1, 2], 2.0
    )
    assert math.isclose(margin_val, 23.0)
    print(f"[PASS] Gen 62 Disk Skill Library         : {len(present)}/6 Primitives Verified (stabilized_route_margin={margin_val})")

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

    # 4. Gen 51-53 Autonomous Causal Program Slicing (delta_slice) check
    broken_repo = {
        "pkg_normalizer": "from stdlib_primitives import kahan_max_subarray\ndef peak_signal(series: list[float], scale: float) -> float:\n    return float(kahan_max_subarray(series))\n",
        "pkg_scorer": "from stdlib_primitives import weighted_mismatch_cost, widest_path_capacity\ndef route_health_score(adj: dict, route: list[int], s1: str, s2: str, weights: list[float], scale: float = 1.0) -> float:\n    return float((widest_path_capacity(adj, route) - weighted_mismatch_cost(s1, s2, weights)) * float(scale))\n",
    }
    contracts = [
        ("pkg_normalizer", lambda e: math.isclose(e["pkg_normalizer"].peak_signal([1.0, 3.0], 2.0), 8.0)),
        ("pkg_scorer", lambda e: math.isclose(e["pkg_scorer"].route_health_score({0: {1: 10.0}, 1: {2: 8.0}}, [0, 1, 2], "1010", "1110", [1.0, 2.0, 1.0, 1.0], 2.0), 12.0)),
    ]
    d_slice = daemon.bisect_repository_contracts(broken_repo, contracts, {"stdlib_primitives": stdlib_primitives})
    assert d_slice == {"pkg_normalizer": 1, "pkg_scorer": 0}
    print(f"[PASS] Gen 52 Autonomous Causal Slicer   : delta_slice={d_slice} (Spared pkg_scorer)")

    # 5. Gen 58-60 ReLoRA & Disaggregated Queue check
    relora_res = ReLoRATrunkConsolidator().consolidate_and_reset_tangent_space(num_layers=196, d_out=256, d_in=256, rank=16)
    disagg_res = DisaggregatedActorLearnerEngine().stream_actor_rollouts([{"task_id": "verify", "code": comp_code, "tokens": 46, "reward": 1.12}])
    assert relora_res["lora_b_zero_reset_verified"] and disagg_res["vram_footprint_reduction_pct"] > 99.0
    print(f"[PASS] Gen 58-60 ReLoRA & Disaggregation : Drift={relora_res['max_forward_equivalence_drift']:.2e} | VRAM Saved={disagg_res['vram_footprint_reduction_pct']}%")

    # 6. Verify v8 & v9 Live GPU Self-Modifying Telemetry
    assert (ROOT / "v8_complete_rsi_metrics.json").exists(), "Missing v8_complete_rsi_metrics.json"
    assert (ROOT / "v9_unscripted_rsi_metrics.json").exists(), "Missing v9_unscripted_rsi_metrics.json"
    v9 = json.loads((ROOT / "v9_unscripted_rsi_metrics.json").read_text())
    grad_res = v9["gen63_64_live_qwen_orthogonal_gradient_step"]["max_grad_overlap_residual"]
    disk_speedup = v9["gen65_disk_self_hosting_router_patch"]["disk_router_speedup"]
    assert v9["gen65_disk_self_hosting_router_patch"]["committed_to_disk"] is True
    print(f"[PASS] Gen 61-65 Live Self-Modifying RSI : 196-Layer Grad Res={grad_res:.2e} | Disk Router Speedup={disk_speedup}x")

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
    print("======================================================================")
    print(f" ALL V9.0.0 (GEN 1-65) VERIFICATION GATES PASSED IN {elapsed} ms")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
