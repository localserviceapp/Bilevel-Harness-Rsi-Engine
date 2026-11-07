from pathlib import Path
import ast, re, sys, copy, math, time, difflib, signal, textwrap, traceback, collections
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional, Callable

FENCE = chr(96) * 3


class ExecutionTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExecutionTimeoutError("Repository test execution exceeded time limit")


@contextmanager
def _time_limit(seconds: float):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


class BiLevelSWEBenchEvoEngine:
    def __init__(self, router=None, max_repair_turns: int = 4):
        self.router = router
        self.max_repair_turns = max_repair_turns

    @staticmethod
    def localize_fault_ast(repo_files: Dict[str, str], issue_text: str, hint_symbol: Optional[str] = None) -> Dict[str, Any]:
        issue_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", issue_text.lower()))
        best_match = None
        best_score = -1.0

        for file_path, src in repo_files.items():
            try:
                tree = ast.parse(src)
            except Exception:
                continue
            lines = src.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn_name = node.name
                    score = 0.0
                    if hint_symbol and fn_name == hint_symbol:
                        score += 100.0
                    if fn_name.lower() in issue_tokens:
                        score += 30.0
                    if fn_name in issue_text:
                        score += 45.0
                    if file_path.lower() in issue_text.lower():
                        score += 40.0
                    node_src = ast.get_source_segment(src, node) or "\n".join(lines[node.lineno - 1:node.end_lineno])
                    body_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", node_src.lower()))
                    score += len(issue_tokens.intersection(body_tokens)) * 1.5

                    if score > best_score:
                        best_score = score
                        imports = [
                            ast.get_source_segment(src, n) or ""
                            for n in tree.body
                            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                        ]
                        sibling_defs = []
                        for other_fp, other_src in repo_files.items():
                            if other_fp != file_path:
                                sibling_defs.append(f"# Sibling module `{other_fp}` (already loaded in global scope):\n{other_src.strip()}")
                        cross_file_header = ("\n\n".join(sibling_defs + [imp for imp in imports if imp])).strip()

                        best_match = {
                            "file_path": file_path,
                            "symbol_name": fn_name,
                            "lineno": node.lineno,
                            "end_lineno": node.end_lineno,
                            "indent_col": node.col_offset,
                            "import_header": cross_file_header,
                            "target_source": textwrap.dedent(node_src),
                            "full_file_source": src,
                        }
        return best_match

    @staticmethod
    def splice_ast_and_build_diff(loc_info: Dict[str, Any], candidate_fn_code: str) -> Tuple[Dict[str, str], str]:
        file_path = loc_info["file_path"]
        orig_src = loc_info["full_file_source"]
        target_name = loc_info["symbol_name"]
        indent_str = " " * loc_info["indent_col"]

        clean_cand = candidate_fn_code.replace("\u00a0", " ").strip()
        extra_top_imports = []
        extracted_fn = textwrap.dedent(clean_cand)

        try:
            cand_tree = ast.parse(textwrap.dedent(clean_cand))
            fn_nodes = [n for n in cand_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            target_node = next((n for n in fn_nodes if n.name == target_name), fn_nodes[-1] if fn_nodes else None)
            if target_node is not None:
                helper_nodes = [
                    n for n in cand_tree.body
                    if n is not target_node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                imp_nodes = [
                    n for n in cand_tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                    and not (isinstance(n, ast.ImportFrom) and n.module and not n.module.startswith(("math", "re", "heapq", "collections", "itertools", "functools", "copy")))
                ]
                extra_top_imports = [ast.unparse(n) for n in imp_nodes]
                combined_fn_nodes = helper_nodes + [target_node]
                extracted_fn = "\n\n".join(ast.unparse(n) for n in combined_fn_nodes)
        except Exception:
            extracted_fn = textwrap.dedent(clean_cand)

        indented_replacement = textwrap.indent(extracted_fn.strip(), indent_str)
        orig_lines = orig_src.splitlines()
        start_idx = loc_info["lineno"] - 1
        end_idx = loc_info["end_lineno"]

        new_lines = extra_top_imports + orig_lines[:start_idx] + indented_replacement.splitlines() + orig_lines[end_idx:]
        new_src = "\n".join(new_lines) + "\n"

        diff_lines = list(
            difflib.unified_diff(
                orig_src.splitlines(keepends=True),
                new_src.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        return {file_path: new_src}, "".join(diff_lines)

    @staticmethod
    def apply_raw_unified_diff(repo_files: Dict[str, str], raw_diff_text: str) -> Optional[Dict[str, str]]:
        patched = dict(repo_files)
        if "--- a/" not in raw_diff_text and "@@" not in raw_diff_text:
            return None
        try:
            current_file = None
            for ln in raw_diff_text.splitlines():
                if ln.startswith("+++ b/"):
                    current_file = ln[6:].strip()
            if not current_file or current_file not in patched:
                return None
            orig_lines = patched[current_file].splitlines()
            hunk_header = re.search(r"@@\s*-(\d+),?(\d*)\s*\+(\d+),?(\d*)\s*@@", raw_diff_text)
            if not hunk_header:
                return None
            start_line = int(hunk_header.group(1)) - 1
            span = int(hunk_header.group(2) or "1")
            added_lines = [ln[1:] for ln in raw_diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
            if not added_lines:
                return None
            new_lines = orig_lines[:start_line] + added_lines + orig_lines[start_line + span:]
            patched[current_file] = "\n".join(new_lines) + "\n"
            ast.parse(patched[current_file])
            return patched
        except Exception:
            return None

    @staticmethod
    def execute_multifile_repo(
        repo_files: Dict[str, str], test_assertions: List[str], target_symbol: str
    ) -> Dict[str, Any]:
        import math, re, heapq, bisect, itertools, functools, collections, copy
        shared_env: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "math": math, "re": re, "heapq": heapq, "bisect": bisect,
            "itertools": itertools, "functools": functools, "collections": collections, "copy": copy,
        }

        for file_path, src in repo_files.items():
            try:
                code_obj = compile(src, file_path, "exec")
                exec(code_obj, shared_env)
            except Exception:
                tb_last = traceback.format_exc().splitlines()[-1]
                return {
                    "passed": False,
                    "passed_count": 0,
                    "tier": "TIER1_REPO_COMPILE",
                    "diagnostic": f"Syntax/Import error in `{file_path}`: {tb_last}",
                }

        passed_cnt = 0
        for assert_stmt in test_assertions:
            try:
                with _time_limit(0.35):
                    exec(compile(assert_stmt, "<repo_test>", "exec"), shared_env)
                passed_cnt += 1
            except Exception as exc:
                captured_locals = {}

                def tracer(frame, event, arg):
                    if frame.f_code.co_name == target_symbol and event in ("return", "exception", "line"):
                        snap = {
                            k: repr(v)[:65]
                            for k, v in list(frame.f_locals.items())[:6]
                            if not k.startswith("__") and not callable(v)
                        }
                        if snap:
                            captured_locals["state"] = snap
                    return tracer

                try:
                    with _time_limit(0.20):
                        sys.settrace(tracer)
                        try:
                            exec(compile(assert_stmt, "<repo_trace>", "exec"), shared_env)
                        except Exception:
                            pass
                        finally:
                            sys.settrace(None)
                except Exception:
                    sys.settrace(None)

                slice_str = f"Causal Slice in `{target_symbol}`: {captured_locals.get('state', '<none>')}"
                return {
                    "passed": False,
                    "passed_count": passed_cnt,
                    "tier": "TIER2_TEST_FAIL",
                    "diagnostic": f"Test `{assert_stmt}` failed with `{type(exc).__name__}: {exc}`. {slice_str}",
                }

        return {
            "passed": True,
            "passed_count": passed_cnt,
            "tier": "PASS",
            "diagnostic": "PASS",
        }

    def solve_repository_task(
        self,
        task: Dict[str, Any],
        generate_fn: Callable,
        batch_generate_fn: Callable,
    ) -> Dict[str, Any]:
        repo_files = copy.deepcopy(task["repo_files"])
        issue_text = task["issue_description"]
        public_repro_tests = list(task["public_repro_assertions"])
        loc = self.localize_fault_ast(repo_files, issue_text, task.get("hint_symbol"))

        target_fn = loc["symbol_name"]
        file_path = loc["file_path"]
        orig_fn_src = loc["target_source"]
        import_hdr = loc["import_header"]

        total_tokens = 0
        total_ms = 0.0

        p_direct = (
            f"You are fixing `{target_fn}` in `{file_path}` inside a multi-file Python repository.\n"
            f"IMPORTANT: Helper functions and classes shown in Cross-File Context are ALREADY loaded in global scope (call them directly; do NOT import non-existent local modules).\n"
            f"Issue Specification:\n{issue_text}\n\n"
            f"Cross-File Context (Already in Scope):\n{FENCE}python\n{import_hdr}\n{FENCE}\n\n"
            f"Current Buggy Implementation:\n{FENCE}python\n{orig_fn_src}\n{FENCE}\n\n"
            f"Reproduction Assertions that MUST pass:\n" + "\n".join(f"- `{a}`" for a in public_repro_tests) + "\n\n"
            f"Return ONLY the complete corrected Python function `def {target_fn}(...)` inside a single {FENCE}python ... {FENCE} block."
        )

        raw_t0, tok_t0, ms_t0 = generate_fn(p_direct, max_new_tokens=350, temperature=0.0)
        total_tokens += tok_t0
        total_ms += ms_t0

        p_map_t0, diff_t0 = self.splice_ast_and_build_diff(loc, raw_t0)
        trial_repo_t0 = {**repo_files, **p_map_t0}
        ver_t0 = self.execute_multifile_repo(trial_repo_t0, public_repro_tests, target_fn)

        if ver_t0["passed"]:
            return {
                "passed_pre_vault": True,
                "turns_used": 0,
                "total_tokens": total_tokens,
                "total_latency_ms": round(total_ms, 2),
                "patched_repo": trial_repo_t0,
                "unified_diff": diff_t0,
                "localized_file": file_path,
                "localized_symbol": target_fn,
            }

        plan_prompt = (
            f"Analyze why `{target_fn}` in `{file_path}` failed verification:\n"
            f"Issue: {issue_text}\n"
            f"Cross-File Context:\n{FENCE}python\n{import_hdr}\n{FENCE}\n"
            f"Diagnostic: {ver_t0['diagnostic']}\n"
            f"Write a concise 3-bullet fix checklist."
        )
        s2_plan, tok_p, ms_p = generate_fn(plan_prompt, max_new_tokens=160, temperature=0.0)
        total_tokens += tok_p
        total_ms += ms_p

        p_guided = (
            f"Fix `{target_fn}` in `{file_path}` following this checklist:\n{s2_plan}\n\n"
            f"Cross-File Context (Already in Scope — do NOT import local files):\n{FENCE}python\n{import_hdr}\n{FENCE}\n\n"
            f"Issue:\n{issue_text}\n\n"
            f"Original Function:\n{FENCE}python\n{orig_fn_src}\n{FENCE}\n\n"
            f"Return ONLY the corrected function `def {target_fn}(...)` inside a single {FENCE}python ... {FENCE} block."
        )

        raw_a, tok_a, ms_a = generate_fn(p_guided, max_new_tokens=360, temperature=0.0)
        batch_b, tok_b, ms_b = batch_generate_fn(p_guided, max_new_tokens=360, temperature=0.40, num_samples=4)
        total_tokens += (tok_a + tok_b)
        total_ms += (ms_a + ms_b)

        best_repo = trial_repo_t0
        best_diff = diff_t0
        best_ver = ver_t0
        best_cand_fn = raw_t0

        for cand_str in [raw_a] + batch_b:
            p_map, p_diff = self.splice_ast_and_build_diff(loc, cand_str)
            t_repo = {**repo_files, **p_map}
            v = self.execute_multifile_repo(t_repo, public_repro_tests, target_fn)
            if v["passed"]:
                return {
                    "passed_pre_vault": True,
                    "turns_used": 1,
                    "total_tokens": total_tokens,
                    "total_latency_ms": round(total_ms, 2),
                    "patched_repo": t_repo,
                    "unified_diff": p_diff,
                    "localized_file": file_path,
                    "localized_symbol": target_fn,
                }
            if v["passed_count"] >= best_ver["passed_count"]:
                best_repo, best_diff, best_ver, best_cand_fn = t_repo, p_diff, v, cand_str

        temp_sched = [0.0, 0.30, 0.55, 0.75]
        for turn in range(1, self.max_repair_turns + 1):
            rep_prompt = (
                f"Your patch for `{target_fn}` failed repository verification:\n"
                f"{FENCE}python\n{best_cand_fn.strip()}\n{FENCE}\n"
                f"Diagnostic & Causal Slice: {best_ver['diagnostic']}\n"
                f"Cross-File Context (Already in Global Scope):\n{FENCE}python\n{import_hdr}\n{FENCE}\n"
                f"Issue Specification:\n{issue_text}\n"
                f"Fix the bug and return ONLY `def {target_fn}(...)` inside a single {FENCE}python ... {FENCE} block."
            )
            temp = temp_sched[min(turn - 1, len(temp_sched) - 1)]
            rep_fn, r_tok, r_ms = generate_fn(rep_prompt, max_new_tokens=360, temperature=temp)
            total_tokens += r_tok
            total_ms += r_ms

            p_map, p_diff = self.splice_ast_and_build_diff(loc, rep_fn)
            t_repo = {**repo_files, **p_map}
            v = self.execute_multifile_repo(t_repo, public_repro_tests, target_fn)
            if v["passed"]:
                return {
                    "passed_pre_vault": True,
                    "turns_used": turn + 1,
                    "total_tokens": total_tokens,
                    "total_latency_ms": round(total_ms, 2),
                    "patched_repo": t_repo,
                    "unified_diff": p_diff,
                    "localized_file": file_path,
                    "localized_symbol": target_fn,
                }
            if v["passed_count"] >= best_ver["passed_count"]:
                best_repo, best_diff, best_ver, best_cand_fn = t_repo, p_diff, v, rep_fn

        return {
            "passed_pre_vault": False,
            "turns_used": self.max_repair_turns,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_ms, 2),
            "patched_repo": best_repo,
            "unified_diff": best_diff,
            "localized_file": file_path,
            "localized_symbol": target_fn,
        }
