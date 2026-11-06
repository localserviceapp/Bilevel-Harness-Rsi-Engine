import ast, inspect
from pathlib import Path
from typing import Dict, Any, List

class AutonomousChallenger:
    def inspect_skill_library(self, stdlib_mod) -> List[str]:
        return [name for name, obj in inspect.getmembers(stdlib_mod, inspect.isfunction) if not name.startswith("_")]

    def crystallize_to_disk(self, stdlib_path: str, fn_name: str, fn_code: str) -> Dict[str, Any]:
        p = Path(stdlib_path)
        current_src = p.read_text()
        ast.parse(fn_code)
        if fn_name not in current_src:
            updated_src = current_src.rstrip() + "\n\n\n" + fn_code.strip() + "\n"
            ast.parse(updated_src)
            p.write_text(updated_src)
        final_tree = ast.parse(p.read_text())
        fn_count = sum(1 for n in final_tree.body if isinstance(n, ast.FunctionDef))
        return {"crystallized_function": fn_name, "total_stdlib_primitives_on_disk": fn_count, "stdlib_file": str(p)}
