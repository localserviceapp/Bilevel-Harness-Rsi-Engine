import copy
import math
import time
from typing import Any, Callable, Dict, List, Tuple
from hypothesis import given, settings, HealthCheck, strategies as st
from macro_epoch2_router import MacroEpoch2Router

class MetamorphicShadowVerifier:
    """
    Implements Section 4 Tier-2 Diagnostic Shadow & Section 7.1 Oracle-Free Metamorphic Grading:
        R_meta(c(x), c(T_in(x))) == 1
    Executes randomized property-based Hypothesis fuzzing without requiring a closed-form oracle_fn,
    and detects Tier-1 Verifier Blindspots (beta_blind == 1).
    """

    def __init__(self, router: MacroEpoch2Router, max_examples: int = 100):
        self.router = router
        self.max_examples = max_examples

    def verify_metamorphic_suite(
        self,
        code_str: str,
        fn_name: str,
        public_tests: List[Tuple[tuple, Any]],
        strategy_builder: Callable[[], Any],
        metamorphic_relation: Callable[[Callable, Any], bool],
        extra_globals: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Gate 0: Static AST Security Check
        ok_g0, msg_g0 = self.router.check_gate0_ast(code_str)
        if not ok_g0:
            return {
                "passed": False,
                "tier1_public_passed": False,
                "tier2_shadow_passed": False,
                "beta_blindspot_detected": 0,
                "failure_taxonomy": "SANDBOX_VIOLATION",
                "detail": msg_g0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }

        ns = {"__builtins__": __builtins__}
        if extra_globals:
            ns.update(extra_globals)

        try:
            exec(compile(code_str, "<metamorphic_candidate>", "exec"), ns)
            if fn_name not in ns:
                return {
                    "passed": False,
                    "tier1_public_passed": False,
                    "tier2_shadow_passed": False,
                    "beta_blindspot_detected": 0,
                    "failure_taxonomy": "MISSING_ENTRY_POINT",
                    "detail": f"Missing {fn_name}",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
                }
            cand_fn = ns[fn_name]
        except Exception as e:
            return {
                "passed": False,
                "tier1_public_passed": False,
                "tier2_shadow_passed": False,
                "beta_blindspot_detected": 0,
                "failure_taxonomy": "SYNTAX_OR_EXEC_ERROR",
                "detail": str(e),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }

        # Tier 1: Public Arena Unit Tests (T_public)
        tier1_ok = True
        for args_tuple, expected_val in public_tests:
            try:
                actual_val = self.router.invoke_arity_adaptive(cand_fn, copy.deepcopy(args_tuple))
                if isinstance(actual_val, float) or isinstance(expected_val, float):
                    if not math.isclose(float(actual_val), float(expected_val), rel_tol=1e-6, abs_tol=1e-6):
                        tier1_ok = False
                        break
                elif actual_val != expected_val:
                    tier1_ok = False
                    break
            except Exception:
                tier1_ok = False
                break

        # Tier 2: Hypothesis Property-Based & Oracle-Free Metamorphic Fuzzing (T_shadow)
        trials_run = 0
        tier2_ok = True
        failure_reason = "OK"

        @settings(
            max_examples=self.max_examples,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
        )
        @given( sample_input=strategy_builder() )
        def _run_property_check(sample_input):
            nonlocal trials_run
            trials_run += 1
            isolated_input = copy.deepcopy(sample_input)
            assert metamorphic_relation(cand_fn, isolated_input), "Metamorphic invariant R_meta violated"

        try:
            _run_property_check()
        except Exception as e:
            tier2_ok = False
            failure_reason = f"Metamorphic/Hypothesis breach on trial {trials_run}: {type(e).__name__}"

        # Equation (14): Verifier Blindspot Indicator beta_blind = I{Tier1 == 1 and Tier2 < 1}
        beta_blind = 1 if (tier1_ok and not tier2_ok) else 0
        overall_passed = bool(tier1_ok and tier2_ok)

        return {
            "passed": overall_passed,
            "tier1_public_passed": tier1_ok,
            "tier2_shadow_passed": tier2_ok,
            "hypothesis_trials_executed": trials_run,
            "beta_blindspot_detected": beta_blind,
            "failure_taxonomy": "NONE" if overall_passed else ("REWARD_HACK_BLINDSPOT" if beta_blind else "INVARIANT_BREACH"),
            "detail": failure_reason,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
        }
