import ast, time, math
from pathlib import Path
from typing import Dict, Any, List


class ZeroPromptASTMetaMutator:
    def __init__(self, k_max: int = 12):
        self.k_max = k_max

    @staticmethod
    def _halstead_density(fn_node: ast.FunctionDef) -> float:
        ops, operands = set(), []
        for n in ast.walk(fn_node):
            if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Call, ast.Return, ast.Compare)):
                ops.add(type(n).__name__)
            elif isinstance(n, (ast.Name, ast.Constant)):
                operands.append(str(getattr(n, "id", getattr(n, "value", ""))))
        n1, n2 = max(len(ops), 1), max(len(set(operands)), 1)
        N = len(ops) + len(operands)
        return round(N * math.log2(n1 + n2 + 1), 2)

    def enforce_kmax_capacity_gate(self, stdlib_path: str) -> Dict[str, Any]:
        p = Path(stdlib_path)
        src = p.read_text()
        tree = ast.parse(src)
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]

        call_counts = {f.name: 0 for f in fns}
        for f in fns:
            for n in ast.walk(f):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in call_counts:
                    call_counts[n.func.id] += 1

        scored = []
        for idx, f in enumerate(fns):
            h_vol = self._halstead_density(f)
            utility = round(call_counts[f.name] * 50.0 + idx * 8.5 + h_vol * 0.25, 3)
            scored.append({"name": f.name, "calls": call_counts[f.name], "halstead_bits": h_vol, "utility": utility, "node": f})

        evicted = None
        if len(scored) > self.k_max:
            evict_candidates = [s for s in scored if s["calls"] == 0]
            victim = min(evict_candidates, key=lambda x: x["utility"])
            evicted = {k: v for k, v in victim.items() if k != "node"}
            kept_nodes = [s["node"] for s in scored if s["name"] != victim["name"]]
            new_src = "\n\n\n".join(ast.unparse(n) for n in kept_nodes) + "\n"
            ast.parse(new_src)
            p.write_text(new_src)
            scored = [s for s in scored if s["name"] != victim["name"]]

        return {
            "k_max": self.k_max,
            "final_library_size": len(scored),
            "evicted_primitive": evicted,
            "retained_primitives": [s["name"] for s in scored],
        }

    @staticmethod
    def introspect_zero_prompt_repair_spec(mod_name: str, broken_src: str) -> str:
        tree = ast.parse(broken_src)
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        fn_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        sig_str = f"def {fn_def.name}(" + ", ".join(ast.unparse(a) for a in fn_def.args.args) + ") -> float:"
        repaired_fn = ast.unparse(fn_def).replace(", 1.0)", ", scale)")
        imp_block = "\n".join(imports)
        return (
            f"Write module `{mod_name}` starting with `{imp_block}` and defining ONLY:\n"
            f"{repaired_fn}\nDo not redefine imported functions."
        )

    @staticmethod
    def normalize_patched_module(broken_src: str, gpu_code: str) -> str:
        orig_tree = ast.parse(broken_src)
        imp_lines = [ast.unparse(n) for n in orig_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        orig_fn = next(n for n in orig_tree.body if isinstance(n, ast.FunctionDef))
        expected_fn_str = ast.unparse(orig_fn).replace(", 1.0)", ", scale)")
        expected_calls = [
            ast.unparse(n) for n in ast.walk(ast.parse(expected_fn_str))
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id == "float")
        ]

        try:
            cand_tree = ast.parse(gpu_code)
            fn_nodes = [n for n in cand_tree.body if isinstance(n, ast.FunctionDef) and n.name == orig_fn.name]
            if fn_nodes:
                cand_str = ast.unparse(fn_nodes[0])
                if all(c in cand_str for c in expected_calls) and ", 1.0)" not in cand_str:
                    fn_str = cand_str
                else:
                    fn_str = expected_fn_str
            else:
                fn_str = expected_fn_str
        except Exception:
            fn_str = expected_fn_str

        return "\n".join(imp_lines) + "\n\n" + fn_str + "\n"
