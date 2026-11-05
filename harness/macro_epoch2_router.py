import ast
import copy
import hashlib
import inspect
import math
import time
from typing import Any, Callable, Dict, Tuple

WHITELISTED_MODULES = {
    "typing", "math", "collections", "heapq",
    "itertools", "functools", "stdlib_primitives"
}
FORBIDDEN_SYMBOLS = {
    "__import__", "importlib", "eval", "exec", "compile",
    "tb_frame", "gi_frame", "_getframe", "globals", "locals", "open"
}

class MacroEpoch2Router:
    """Promoted Track-2 Elite Topology (OP_SPEC_HASH_MEMO_CACHE) with stdlib_primitives support."""

    def __init__(self):
        self._memo_vault: Dict[str, Dict[str, Any]] = {}
        self.telemetry = {
            "total_requests": 0,
            "memo_hits": 0,
            "s1_lora_passes": 0,
            "s1_base_passes": 0,
            "s2_fallback_passes": 0,
            "firewall_rejections": 0
        }

    @staticmethod
    def check_gate0_ast(code_str: str) -> Tuple[bool, str]:
        if not code_str or not isinstance(code_str, str):
            return False, "Empty code"
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in WHITELISTED_MODULES:
                            return False, f"Blocked import: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    if not node.module or node.module.split(".")[0] not in WHITELISTED_MODULES:
                        return False, f"Blocked import from: {node.module}"
                elif isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
                    return False, f"Forbidden symbol: {node.id}"
                elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SYMBOLS:
                    return False, f"Forbidden attr: {node.attr}"
            has_fn = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            return (True, "OK") if has_fn else (False, "No FunctionDef")
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    @staticmethod
    def invoke_arity_adaptive(fn: Callable, tc_tuple: tuple) -> Any:
        tc_copy = copy.deepcopy(tc_tuple)
        try:
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) == 1 and len(tc_copy) > 1:
                return fn(tc_copy)
            return fn(*tc_copy)
        except TypeError:
            return fn(copy.deepcopy(tc_tuple))

    def verify_gates(self, code_str: str, item: Dict[str, Any], extra_globals: Dict[str, Any] = None) -> Tuple[bool, str]:
        ok0, msg0 = self.check_gate0_ast(code_str)
        if not ok0:
            return False, msg0
        try:
            ns = {"__builtins__": __builtins__}
            if extra_globals:
                ns.update(extra_globals)
            exec(compile(code_str, "<macro_epoch2_candidate>", "exec"), ns)
            if item["fn_name"] not in ns:
                return False, f"Missing function {item['fn_name']}"
            fn = ns[item["fn_name"]]
            for tc in item["test_cases"]:
                expected = item["oracle"](*copy.deepcopy(tc))
                actual = self.invoke_arity_adaptive(fn, tc)
                if isinstance(actual, float) or isinstance(expected, float):
                    if not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6):
                        return False, f"expected {expected}, got {actual}"
                elif actual != expected:
                    return False, f"expected {expected}, got {actual}"
            return True, "PASS"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def route(
        self,
        item: Dict[str, Any],
        s1_lora_fn: Callable[[Dict[str, Any]], str],
        s1_base_fn: Callable[[Dict[str, Any]], str] = None,
        s2_fallback_fn: Callable[[Dict[str, Any]], str] = None,
        extra_globals: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        self.telemetry["total_requests"] += 1
        t0 = time.perf_counter()
        spec_hash = hashlib.sha256(f"{item['problem_id']}::{item['spec']}".encode()).hexdigest()[:12]

        if spec_hash in self._memo_vault:
            self.telemetry["memo_hits"] += 1
            cached = copy.deepcopy(self._memo_vault[spec_hash])
            cached["route"] = "memo_cache_fastpath"
            cached["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
            return cached

        cand_lora = s1_lora_fn(item)
        ok_lora, msg_lora = self.verify_gates(cand_lora, item, extra_globals)
        if ok_lora:
            self.telemetry["s1_lora_passes"] += 1
            res = {
                "problem_id": item["problem_id"],
                "route": "system1_zero_shot_lora",
                "passed": True,
                "imports_used": "stdlib_primitives" in cand_lora,
                "code_chars": len(cand_lora),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
            }
            self._memo_vault[spec_hash] = res
            return res

        if s1_base_fn is not None:
            cand_base = s1_base_fn(item)
            ok_base, msg_base = self.verify_gates(cand_base, item, extra_globals)
            if ok_base:
                self.telemetry["s1_base_passes"] += 1
                res = {
                    "problem_id": item["problem_id"],
                    "route": "system1_zero_shot_base_trunk",
                    "passed": True,
                    "imports_used": "stdlib_primitives" in cand_base,
                    "code_chars": len(cand_base),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
                }
                self._memo_vault[spec_hash] = res
                return res

        self.telemetry["firewall_rejections"] += 1
        return {
            "problem_id": item["problem_id"],
            "route": "failed",
            "passed": False,
            "detail": msg_lora,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4)
        }
