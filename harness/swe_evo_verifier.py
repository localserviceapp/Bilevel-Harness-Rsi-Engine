import ast
import copy
import math
import sys
import time
import types
from typing import Any, Callable, Dict, List, Set, Tuple
from hypothesis import given, settings, HealthCheck
from macro_epoch2_router import MacroEpoch2Router, WHITELISTED_MODULES, FORBIDDEN_SYMBOLS

class SweEvoMultiFileVerifier:
    """
    Step 4: Multi-File Repository Evolution (SWE-EVO) Transactional Verifier.
    Enforces cross-module Gate-0 AST security, verifies acyclic internal module dependencies,
    measures multi-file Specification Mutation Elasticity (Delta_AST, Eq. 5),
    and executes atomic transactional shadow verification with automatic rollback.
    """

    def __init__(self, max_examples: int = 100):
        self.max_examples = max_examples

    @classmethod
    def compute_multi_file_delta_ast(cls, base_files: Dict[str, str], patched_files: Dict[str, str]) -> float:
        """Computes normalized AST node structural delta (Eq. 5 proxy) across a multi-file patch."""
        total_base_nodes = 0
        total_diff_nodes = 0
        for fname, base_code in base_files.items():
            patch_code = patched_files.get(fname, base_code)
            b_seq = [type(n).__name__ for n in ast.walk(ast.parse(base_code))]
            p_seq = [type(n).__name__ for n in ast.walk(ast.parse(patch_code))]
            total_base_nodes += max(1, len(b_seq))
            common = sum(1 for a, b in zip(b_seq, p_seq) if a == b)
            diff = max(len(b_seq), len(p_seq)) - common
            total_diff_nodes += diff
        return round(min(1.0, total_diff_nodes / max(1, total_base_nodes)), 4)

    @staticmethod
    def verify_multi_file_gate0(
        patched_files: Dict[str, str],
        allowed_internal_modules: Set[str]
    ) -> Tuple[bool, str, Dict[str, List[str]]]:
        dep_graph: Dict[str, List[str]] = {m: [] for m in allowed_internal_modules}
        permitted = set(WHITELISTED_MODULES) | allowed_internal_modules | {"stdlib_primitives"}

        for mod_name, code_str in patched_files.items():
            try:
                tree = ast.parse(code_str)
            except SyntaxError as e:
                return False, f"SyntaxError in {mod_name}: {e}", dep_graph

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root not in permitted:
                            return False, f"Blocked import '{root}' in {mod_name}", dep_graph
                        if root in allowed_internal_modules:
                            dep_graph[mod_name].append(root)
                elif isinstance(node, ast.ImportFrom):
                    if not node.module:
                        return False, f"Relative import blocked in {mod_name}", dep_graph
                    root = node.module.split(".")[0]
                    if root not in permitted:
                        return False, f"Blocked import from '{root}' in {mod_name}", dep_graph
                    if root in allowed_internal_modules:
                        dep_graph[mod_name].append(root)
                elif isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
                    return False, f"Forbidden symbol '{node.id}' in {mod_name}", dep_graph
                elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SYMBOLS:
                    return False, f"Forbidden attribute '{node.attr}' in {mod_name}", dep_graph

        visited: Dict[str, int] = {m: 0 for m in allowed_internal_modules}
        def has_cycle(u: str) -> bool:
            visited[u] = 1
            for v in dep_graph.get(u, []):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and has_cycle(v):
                    return True
            visited[u] = 2
            return False

        for mod in allowed_internal_modules:
            if visited[mod] == 0 and has_cycle(mod):
                return False, "Circular dependency detected in multi-file patch", dep_graph

        return True, "OK", dep_graph

    def evaluate_patch_transaction(
        self,
        patch_id: str,
        base_repo: Dict[str, str],
        candidate_patch: Dict[str, str],
        entry_module: str,
        entry_fn: str,
        public_tests: List[Tuple[tuple, Any]],
        strategy_builder: Callable[[], Any],
        metamorphic_relation: Callable[[Callable, Any], bool],
        extra_modules: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        modified_list = list(candidate_patch.keys())
        snapshot_repo = copy.deepcopy(base_repo)
        merged_repo = copy.deepcopy(base_repo)
        merged_repo.update(candidate_patch)

        internal_mods = set(merged_repo.keys())
        ok_g0, msg_g0, dep_graph = self.verify_multi_file_gate0(merged_repo, internal_mods)
        if not ok_g0:
            return {
                "patch_id": patch_id,
                "files_modified": modified_list,
                "committed": False,
                "rolled_back": True,
                "gate0_passed": False,
                "tier1_passed": False,
                "tier2_passed": False,
                "hypothesis_trials": 0,
                "delta_ast": 1.0,
                "detail": msg_g0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }

        # Save previous sys.modules state for atomic rollback after transaction
        saved_sys_modules = {m: sys.modules.get(m) for m in internal_mods}
        loaded_mods: Dict[str, types.ModuleType] = {}

        try:
            if extra_modules:
                for k, v in extra_modules.items():
                    sys.modules[k] = v
                    loaded_mods[k] = v

            load_order = sorted(internal_mods, key=lambda m: len(dep_graph.get(m, [])))
            for mod_name in load_order:
                mod_obj = types.ModuleType(mod_name)
                mod_obj.__dict__["__builtins__"] = __builtins__
                sys.modules[mod_name] = mod_obj
                exec(compile(merged_repo[mod_name], f"<swe_evo:{mod_name}>", "exec"), mod_obj.__dict__)
                loaded_mods[mod_name] = mod_obj

            target_callable = getattr(loaded_mods[entry_module], entry_fn)

            # Tier 1: Public Integration Tests
            tier1_ok = True
            for args_tuple, expected_val in public_tests:
                try:
                    actual_val = MacroEpoch2Router.invoke_arity_adaptive(target_callable, copy.deepcopy(args_tuple))
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

            # Tier 2: Hypothesis Multi-File Metamorphic Verification
            trials_run = 0
            tier2_ok = True
            detail_msg = "COMMITTED"

            @settings(
                max_examples=self.max_examples,
                deadline=None,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
            )
            @given(sample_input=strategy_builder())
            def _fuzz_pipeline(sample_input):
                nonlocal trials_run
                trials_run += 1
                assert metamorphic_relation(target_callable, copy.deepcopy(sample_input))

            try:
                _fuzz_pipeline()
            except Exception as e:
                tier2_ok = False
                detail_msg = f"Rolled back on Hypothesis trial {trials_run}: {type(e).__name__}"

            committed = bool(tier1_ok and tier2_ok)
            delta_ast = self.compute_multi_file_delta_ast(snapshot_repo, merged_repo) if committed else 1.0

            return {
                "patch_id": patch_id,
                "files_modified": modified_list,
                "committed": committed,
                "rolled_back": not committed,
                "gate0_passed": True,
                "tier1_passed": tier1_ok,
                "tier2_passed": tier2_ok,
                "hypothesis_trials": trials_run,
                "delta_ast": delta_ast,
                "detail": detail_msg if not committed else "COMMITTED",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }
        except Exception as e:
            return {
                "patch_id": patch_id,
                "files_modified": modified_list,
                "committed": False,
                "rolled_back": True,
                "gate0_passed": True,
                "tier1_passed": False,
                "tier2_passed": False,
                "hypothesis_trials": 0,
                "delta_ast": 1.0,
                "detail": f"ModuleLinkError: {e}",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }
        finally:
            for m, prev in saved_sys_modules.items():
                if prev is None:
                    sys.modules.pop(m, None)
                else:
                    sys.modules[m] = prev
