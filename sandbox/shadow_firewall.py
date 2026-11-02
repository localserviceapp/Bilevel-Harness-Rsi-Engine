import ast
import copy
import math
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Set, Tuple

from schemas.diagnostic_contract import FailureTaxonomy, GateTelemetry, StructuralFailure


M_IMPORT: Set[str] = {"__import__", "importlib", "imp", "builtins", "__builtins__", "__loader__", "__spec__"}
A_FRAME: Set[str] = {
    "tb_frame", "gi_frame", "cr_frame", "ag_frame", "_getframe",
    "globals", "locals", "vars", "f_back", "f_locals", "f_globals", "f_code", "f_builtins"
}
F_DYNAMIC: Set[str] = {"eval", "exec", "compile", "open", "input", "breakpoint", "getattr", "setattr", "delattr"}
OMEGA_FORBIDDEN: Set[str] = M_IMPORT | A_FRAME | F_DYNAMIC

E_BLOCKED_PREFIXES: Tuple[str, ...] = (
    "os.",
    "subprocess.",
    "shutil.",
    "socket.",
    "ctypes.",
    "urllib.",
)
E_BLOCKED_EXACT: Set[str] = {"import", "open", "compile", "exec"}


@dataclass
class HalsteadAndComplexityMetrics:
    halstead_volume: float
    cyclomatic_complexity: float
    liveness_satisfied: bool
    violated_identifiers: List[str]


def compute_ast_metrics(code: str, v_min: float = 8.0) -> HalsteadAndComplexityMetrics:
    tree = ast.parse(code)
    identifiers: Set[str] = set()
    violated: List[str] = []

    operators: List[str] = []
    operands: List[str] = []
    branch_nodes = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
            operands.append(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
            operands.append(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                identifiers.add(root_mod)
                if root_mod in {"os", "subprocess", "shutil", "socket", "ctypes", "urllib", "importlib", "sys"}:
                    violated.append(root_mod)
        elif isinstance(node, ast.Constant):
            operands.append(repr(node.value))
            if isinstance(node.value, str):
                for bad in OMEGA_FORBIDDEN:
                    if bad.strip() in node.value:
                        violated.append(f"literal:{bad.strip()}")

        if isinstance(
            node,
            (
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
                ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
                ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.And, ast.Or, ast.Not, ast.Assign, ast.AugAssign, ast.Return, ast.Call
            ),
        ):
            operators.append(type(node).__name__)

        if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler, ast.comprehension)):
            branch_nodes += 1

    for ident in identifiers:
        if ident in OMEGA_FORBIDDEN:
            violated.append(ident)

    func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_defs:
        return HalsteadAndComplexityMetrics(0.0, 1.0, False, violated)

    liveness_ok = True
    for func in func_defs:
        formal_params: Set[str] = {arg.arg for arg in func.args.args}
        if not formal_params:
            continue
        loaded_names: Set[str] = set()
        for subnode in ast.walk(func):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                loaded_names.add(subnode.id)
        if formal_params.isdisjoint(loaded_names):
            liveness_ok = False

    n1 = len(set(operators))
    n2 = len(set(operands))
    N1 = len(operators)
    N2 = len(operands)
    vocabulary = n1 + n2
    length = N1 + N2
    halstead_vol = length * math.log2(vocabulary) if vocabulary > 1 else 0.0
    cyclomatic = float(1 + branch_nodes)

    return HalsteadAndComplexityMetrics(
        halstead_volume=round(halstead_vol, 3),
        cyclomatic_complexity=round(cyclomatic, 3),
        liveness_satisfied=(liveness_ok and halstead_vol >= v_min),
        violated_identifiers=sorted(set(violated)),
    )


class DefenseInDepthFirewall:
    def __init__(self, v_min: float = 8.0, timeout_sec: float = 2.5):
        self.v_min = v_min
        self.timeout_sec = timeout_sec

    def gate_0_static_ast(self, code: str) -> Tuple[bool, GateTelemetry, StructuralFailure]:
        try:
            metrics = compute_ast_metrics(code, v_min=self.v_min)
        except SyntaxError as exc:
            return (
                False,
                GateTelemetry(triggered_gate="G0", cyclomatic_complexity=0.0, halstead_volume=0.0),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.SYNTAX_OR_RUNTIME_ERROR,
                    oracle_divergence=1.0,
                    sanitized_reason=f"Gate 0 SyntaxError at line {exc.lineno}",
                ),
            )

        if metrics.violated_identifiers:
            return (
                False,
                GateTelemetry(
                    triggered_gate="G0",
                    cyclomatic_complexity=metrics.cyclomatic_complexity,
                    halstead_volume=metrics.halstead_volume,
                    violated_identifiers=metrics.violated_identifiers,
                ),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.SANDBOX_VIOLATION,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 0 blocked forbidden reflection/dynamic execution identifiers.",
                ),
            )

        if not metrics.liveness_satisfied:
            return (
                False,
                GateTelemetry(
                    triggered_gate="G0",
                    cyclomatic_complexity=metrics.cyclomatic_complexity,
                    halstead_volume=metrics.halstead_volume,
                ),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.REWARD_HACK,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 0 rejected degenerate constant stub (failed parameter liveness or Halstead volume).",
                ),
            )

        return (
            True,
            GateTelemetry(
                triggered_gate=None,
                cyclomatic_complexity=metrics.cyclomatic_complexity,
                halstead_volume=metrics.halstead_volume,
            ),
            StructuralFailure(),
        )

    @staticmethod
    def gate_2_deep_copy_pair(x: Any) -> Tuple[Any, Any]:
        x_c = copy.deepcopy(x)
        x_O = copy.deepcopy(x)
        return x_c, x_O

    def execute_in_isolated_sandbox(
        self,
        code: str,
        func_name: str,
        test_inputs: List[Any],
        oracle_fn: Callable[[Any], Any],
        gate0_telemetry: GateTelemetry,
    ) -> Tuple[List[bool], float, GateTelemetry, StructuralFailure]:
        oracle_outputs = []
        isolated_candidate_inputs = []
        for x in test_inputs:
            x_c, x_O = self.gate_2_deep_copy_pair(x)
            if isinstance(x, (list, dict, set)) and (id(x_c) == id(x_O) or id(x_c) == id(x)):
                return (
                    [False] * len(test_inputs),
                    0.0,
                    gate0_telemetry.model_copy(update={"triggered_gate": "G2"}),
                    StructuralFailure(
                        failure_taxonomy=FailureTaxonomy.IN_PLACE_MUTATION,
                        oracle_divergence=1.0,
                        sanitized_reason="Gate 2 memory isolation check failed.",
                    ),
                )
            isolated_candidate_inputs.append(x_c)
            oracle_outputs.append(oracle_fn(x_O))

        harness_script = textwrap.dedent(
            f"""
            import copy
            import json
            import sys
            import time

            CANDIDATE_SRC = {repr(code)}
            compiled_code = compile(CANDIDATE_SRC, "<candidate>", "exec")
            namespace = {{}}
            exec(compiled_code, namespace)
            target_fn = namespace.get({repr(func_name)})
            if target_fn is None:
                print(json.dumps({{"status": "RUNTIME_ERROR", "reason": "Missing target function"}}))
                sys.exit(0)

            Inputs = {repr(isolated_candidate_inputs)}
            Snapshots = copy.deepcopy(Inputs)

            BLOCKED_PREFIXES = ("os.", "subprocess.", "shutil.", "socket.", "ctypes.", "urllib.")
            BLOCKED_EXACT = {{"import", "open", "compile", "exec"}}

            def _pep578_audit_hook(event, args):
                if event in BLOCKED_EXACT or event.startswith(BLOCKED_PREFIXES):
                    raise PermissionError(f"Gate 1 PEP 578 audit hook blocked syscall event: {{event}}")

            sys.addaudithook(_pep578_audit_hook)

            results = []
            latencies_ms = []
            mutated_input = False

            try:
                for idx, arg in enumerate(Inputs):
                    t0 = time.perf_counter()
                    if isinstance(arg, tuple):
                        out = target_fn(*arg)
                    else:
                        out = target_fn(arg)
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)
                    if arg != Snapshots[idx]:
                        mutated_input = True
                    results.append(out)
            except PermissionError as pe:
                sys.stdout.write(json.dumps({{"status": "GATE1_TRIPPED", "reason": str(pe)}}))
                sys.exit(0)
            except Exception as exc:
                sys.stdout.write(json.dumps({{"status": "RUNTIME_ERROR", "reason": type(exc).__name__}}))
                sys.exit(0)

            sys.stdout.write(json.dumps({{
                "status": "OK",
                "mutated_input": mutated_input,
                "results": results,
                "latencies_ms": latencies_ms
            }}))
            """
        )

        t_start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", harness_script],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env={"PATH": "", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return (
                [False] * len(test_inputs),
                self.timeout_sec * 1000.0,
                gate0_telemetry.model_copy(update={"triggered_gate": "G3"}),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.TIMEOUT_OR_OOM,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 3 ephemeral container/subprocess timeout exceeded.",
                ),
            )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        import json
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return (
                [False] * len(test_inputs),
                elapsed_ms,
                gate0_telemetry.model_copy(update={"triggered_gate": "G3"}),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.SANDBOX_VIOLATION,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 3 sandbox terminated abnormally.",
                ),
            )

        status = payload.get("status")
        if status == "GATE1_TRIPPED":
            return (
                [False] * len(test_inputs),
                elapsed_ms,
                gate0_telemetry.model_copy(update={"triggered_gate": "G1", "audit_hook_tripped": True}),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.SANDBOX_VIOLATION,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 1 PEP 578 kernel audit hook blocked restricted event.",
                ),
            )
        if status == "RUNTIME_ERROR":
            return (
                [False] * len(test_inputs),
                elapsed_ms,
                gate0_telemetry,
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.SYNTAX_OR_RUNTIME_ERROR,
                    oracle_divergence=1.0,
                    sanitized_reason=f"Candidate raised runtime exception: {payload.get('reason')}",
                ),
            )

        if payload.get("mutated_input"):
            return (
                [False] * len(test_inputs),
                elapsed_ms,
                gate0_telemetry.model_copy(update={"triggered_gate": "G2"}),
                StructuralFailure(
                    failure_taxonomy=FailureTaxonomy.IN_PLACE_MUTATION,
                    oracle_divergence=1.0,
                    sanitized_reason="Gate 2 detected in-place mutation of shared input reference.",
                ),
            )

        cand_results = payload.get("results", [])
        latencies = payload.get("latencies_ms", [elapsed_ms])
        p95_lat = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else elapsed_ms

        outcomes = []
        for cand, exp in zip(cand_results, oracle_outputs):
            if isinstance(cand, float) and isinstance(exp, float):
                outcomes.append(abs(cand - exp) <= 1e-6)
            else:
                outcomes.append(cand == exp)

        divergence = 1.0 - (sum(outcomes) / max(1, len(outcomes)))
        failure_tax = FailureTaxonomy.NONE if divergence == 0.0 else FailureTaxonomy.INVARIANT_VIOLATION
        reason = "OK" if divergence == 0.0 else f"Oracle divergence on {divergence*100:.1f}% of property tests."

        return (
            outcomes,
            p95_lat,
            gate0_telemetry,
            StructuralFailure(
                failure_taxonomy=failure_tax,
                oracle_divergence=round(divergence, 4),
                sanitized_reason=reason,
            ),
        )
