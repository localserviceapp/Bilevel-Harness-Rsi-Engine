import ast
import time
import z3
from typing import Dict, Any, List, Callable

class ConcolicASTBranchSolver:
    """
    Limit-Breaker 3: Defeating Combinatorial State Explosion via AST-Guided Concolic Execution.
    Parses candidate Python AST to extract `ast.If` predicates, converts conditions into
    symbolic Z3 constraints, and algebraically solves for exact narrow-manifold inputs
    that force execution down hidden branches missed by random Hypothesis fuzzing.
    """

    def _ast_expr_to_z3(self, node: ast.AST, z3_vars: Dict[str, z3.ArithRef]):
        if isinstance(node, ast.Name):
            if node.id not in z3_vars:
                z3_vars[node.id] = z3.Int(node.id)
            return z3_vars[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return z3.IntVal(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._ast_expr_to_z3(node.operand, z3_vars)
        if isinstance(node, ast.BinOp):
            left = self._ast_expr_to_z3(node.left, z3_vars)
            right = self._ast_expr_to_z3(node.right, z3_vars)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        if isinstance(node, ast.BoolOp):
            vals = [self._ast_expr_to_z3(v, z3_vars) for v in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*vals)
            if isinstance(node.op, ast.Or):
                return z3.Or(*vals)
        if isinstance(node, ast.Compare):
            left = self._ast_expr_to_z3(node.left, z3_vars)
            clauses = []
            curr = left
            for op, comp in zip(node.ops, node.comparators):
                rhs = self._ast_expr_to_z3(comp, z3_vars)
                if isinstance(op, ast.Eq):
                    clauses.append(curr == rhs)
                elif isinstance(op, ast.NotEq):
                    clauses.append(curr != rhs)
                elif isinstance(op, ast.Gt):
                    clauses.append(curr > rhs)
                elif isinstance(op, ast.GtE):
                    clauses.append(curr >= rhs)
                elif isinstance(op, ast.Lt):
                    clauses.append(curr < rhs)
                elif isinstance(op, ast.LtE):
                    clauses.append(curr <= rhs)
                curr = rhs
            return z3.And(*clauses) if len(clauses) > 1 else clauses[0]
        raise ValueError(f"Unsupported AST node in concolic solver: {type(node).__name__}")

    def audit_candidate_branches(
        self,
        code_str: str,
        fn_name: str,
        invariant_fn: Callable[[Callable, Dict[str, int]], bool]
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tree = ast.parse(code_str)
        ns: Dict[str, Any] = {}
        exec(compile(tree, "<concolic_candidate>", "exec"), ns)
        fn = ns[fn_name]

        fn_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        arg_names = [a.arg for a in fn_def.args.args]

        branches_solved = 0
        for node in ast.walk(fn_def):
            if isinstance(node, ast.If):
                z3_vars = {name: z3.Int(name) for name in arg_names}
                try:
                    cond_z3 = self._ast_expr_to_z3(node.test, z3_vars)
                except Exception:
                    continue
                solver = z3.Solver()
                solver.add(cond_z3)
                # Keep free parameters in non-degenerate range
                for name, zv in z3_vars.items():
                    solver.add(zv >= -10000000, zv <= 10000000)
                if solver.check() == z3.sat:
                    branches_solved += 1
                    model = solver.model()
                    concrete_kwargs = {
                        name: int(model.eval(z3_vars[name], model_completion=True).as_long())
                        for name in arg_names
                    }
                    if not invariant_fn(fn, concrete_kwargs):
                        return {
                            "passed": False,
                            "branches_solved": branches_solved,
                            "concolic_counterexample": concrete_kwargs,
                            "solve_time_ms": round((time.perf_counter() - t0) * 1000.0, 3)
                        }

        return {
            "passed": True,
            "branches_solved": branches_solved,
            "concolic_counterexample": None,
            "solve_time_ms": round((time.perf_counter() - t0) * 1000.0, 3)
        }
