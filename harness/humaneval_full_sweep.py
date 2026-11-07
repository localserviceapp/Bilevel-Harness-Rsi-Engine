import ast, re, sys, copy, math, time, random, signal, textwrap, traceback
import heapq, bisect, string, hashlib, operator, itertools, functools, collections, fractions, decimal, statistics, datetime, unicodedata, cmath
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Set, Optional, Union, Callable

TRIPLE_DQ = chr(34) * 3
TRIPLE_SQ = chr(39) * 3
FENCE = chr(96) * 3
INDENT4 = chr(32) * 4

SAFE_IMPORT_NAMES = {
    "math", "re", "copy", "heapq", "bisect", "string", "hashlib", "operator",
    "itertools", "functools", "collections", "fractions", "decimal", "statistics",
    "datetime", "unicodedata", "cmath", "typing", "stdlib_primitives",
}


def build_safe_exec_env(stdlib_module=None) -> Dict[str, Any]:
    env = {
        "math": math, "re": re, "copy": copy, "heapq": heapq, "bisect": bisect,
        "string": string, "hashlib": hashlib, "operator": operator,
        "itertools": itertools, "functools": functools, "collections": collections,
        "fractions": fractions, "decimal": decimal, "statistics": statistics,
        "datetime": datetime, "unicodedata": unicodedata, "cmath": cmath,
        "List": List, "Dict": Dict, "Tuple": Tuple, "Set": Set,
        "Optional": Optional, "Any": Any, "Union": Union, "Callable": Callable,
        "Counter": collections.Counter, "defaultdict": collections.defaultdict,
        "deque": collections.deque, "reduce": functools.reduce,
        "Fraction": fractions.Fraction, "Decimal": decimal.Decimal,
    }
    if stdlib_module is not None:
        env["stdlib_primitives"] = stdlib_module
    return env


class ExecutionTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExecutionTimeoutError("Execution exceeded time limit (infinite loop or exponential recursion)")


@contextmanager
def _time_limit(seconds: float):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


class OptimizedHumanEvalZeroFallbackEngine:
    def __init__(self, router, stdlib_module, max_repair_turns: int = 5):
        self.router = router
        self.stdlib_module = stdlib_module
        self.max_repair_turns = max_repair_turns

    @staticmethod
    def _parse_prompt_ast(prompt_prefix: str) -> ast.Module:
        try:
            return ast.parse(prompt_prefix)
        except Exception:
            return ast.parse(prompt_prefix.rstrip() + "\n" + INDENT4 + "pass\n")

    def check_gate0_allowing_safe_stdlib(self, code_str: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
            filtered_body = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    kept = [alias for alias in node.names if alias.name.split(".")[0] not in SAFE_IMPORT_NAMES]
                    if kept:
                        node.names = kept
                        filtered_body.append(node)
                elif isinstance(node, ast.ImportFrom):
                    mod_root = (node.module or "").split(".")[0]
                    if mod_root not in SAFE_IMPORT_NAMES:
                        filtered_body.append(node)
                else:
                    filtered_body.append(node)
            sanitized_src = ast.unparse(ast.Module(body=filtered_body, type_ignores=[]))
            return self.router.check_gate0_ast(sanitized_src)
        except Exception:
            return self.router.check_gate0_ast(code_str)

    @classmethod
    def build_executable_module(cls, prompt_prefix: str, model_completion: str, entry_point: str) -> str:
        clean_cand = model_completion.replace("\u00a0", " ").strip()
        if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", clean_cand, flags=re.MULTILINE):
            try:
                prefix_tree = cls._parse_prompt_ast(prompt_prefix)
                cand_tree = ast.parse(textwrap.dedent(clean_cand))
                safe_cand_body = [
                    n for n in cand_tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
                ]
                cand_fn_names = {n.name for n in safe_cand_body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                kept_prefix_nodes = [
                    n for n in prefix_tree.body
                    if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in cand_fn_names)
                ]
                prefix_src = ast.unparse(ast.Module(body=kept_prefix_nodes, type_ignores=[])) if kept_prefix_nodes else ""
                cand_src = ast.unparse(ast.Module(body=safe_cand_body, type_ignores=[]))
                return (prefix_src + "\n\n" + cand_src).strip() + "\n"
            except Exception:
                return prompt_prefix + "\n\n" + textwrap.dedent(clean_cand) + "\n"
        else:
            dedented_body = textwrap.dedent(clean_cand)
            indented_body = textwrap.indent(dedented_body, INDENT4)
            return prompt_prefix.rstrip() + "\n" + indented_body + "\n"

    @staticmethod
    def _extract_balanced_call(text: str, entry_point: str) -> Optional[Tuple[str, str]]:
        m = re.search(rf"\b{re.escape(entry_point)}\s*\(", text)
        if not m:
            m_alias = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)
            if not m_alias or m_alias.group(1) in ("for", "if", "while", "print", "assert", "range", "len", "int", "str", "float", "list", "dict", "set", "tuple"):
                return None
            open_paren = m_alias.end() - 1
        else:
            open_paren = m.end() - 1

        depth = 0
        in_str = None
        escape = False
        for idx in range(open_paren, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == chr(92):
                escape = True
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch in (chr(34), chr(39)):
                in_str = ch
                continue
            if ch in ("(", "[", "{"):
                depth += 1
            elif ch in (")", "]", "}"):
                depth -= 1
                if depth == 0:
                    args_slice = text[open_paren:idx + 1]
                    norm_call = f"{entry_point}{args_slice}"
                    return norm_call.strip(), text[idx + 1:].strip()
        return None

    @staticmethod
    def _clean_literal_expectation(raw_exp: str) -> Optional[str]:
        s = raw_exp.replace(TRIPLE_DQ, "").replace(TRIPLE_SQ, "").strip()
        in_str = None
        escape = False
        cut_idx = len(s)
        for idx, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if ch == chr(92):
                escape = True
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch in (chr(34), chr(39)):
                in_str = ch
                continue
            if ch == "#":
                cut_idx = idx
                break
        s = s[:cut_idx].strip().rstrip(".,;")
        if not s:
            return None

        candidates = [s]
        if "=" in s and not s.startswith("="):
            rhs = s.rsplit("=", 1)[-1].strip()
            if rhs:
                candidates.append(rhs)

        for cand in candidates:
            try:
                val = ast.literal_eval(cand)
                if isinstance(val, tuple) and not (cand.startswith("(") and cand.endswith(")")):
                    continue
                return repr(val)
            except Exception:
                continue
        return None

    @classmethod
    def extract_docstring_tests(cls, prompt_prefix: str, entry_point: str) -> List[Tuple[str, str]]:
        tests = []
        raw_lines = [ln.strip() for ln in prompt_prefix.splitlines()]
        i = 0
        while i < len(raw_lines):
            ln = raw_lines[i]
            if ln.startswith(">>>"):
                call_line = ln.lstrip(">").strip()
                while i + 1 < len(raw_lines) and raw_lines[i + 1].startswith("..."):
                    i += 1
                    call_line += " " + raw_lines[i].lstrip(".").strip()
                bal = cls._extract_balanced_call(call_line, entry_point)
                if bal:
                    c_expr, rem = bal
                    if rem and re.match(r"^(?:==|->|=>|➞|=)\s*(.+)$", rem):
                        e_raw = re.sub(r"^(?:==|->|=>|➞|=)\s*", "", rem).strip()
                        e_lit = cls._clean_literal_expectation(e_raw)
                        if e_lit is not None:
                            tests.append((c_expr, e_lit))
                    elif i + 1 < len(raw_lines) and not raw_lines[i + 1].startswith((">>>", TRIPLE_DQ, TRIPLE_SQ)):
                        e_lit = cls._clean_literal_expectation(raw_lines[i + 1])
                        if e_lit is not None:
                            tests.append((c_expr, e_lit))
                            i += 1
            elif not ln.startswith("def ") and ("(" in ln and ")" in ln):
                clean_ln = re.sub(r"^(?:assert|example\s*\d*\s*:|for\s+example\s*:)\s*", "", ln, flags=re.IGNORECASE)
                if i + 1 < len(raw_lines) and (clean_ln.endswith(("->", "=>", "==>", "➞", "=", "[")) or raw_lines[i + 1].startswith(("->", "=>", "==>", "➞", "==", "="))):
                    clean_ln = clean_ln + " " + raw_lines[i + 1].replace(TRIPLE_DQ, "").replace(TRIPLE_SQ, "").strip()
                if entry_point in clean_ln or "==>" in clean_ln or "->" in clean_ln or "=>" in clean_ln or "➞" in clean_ln:
                    bal = cls._extract_balanced_call(clean_ln, entry_point)
                    if bal:
                        c_expr, rem = bal
                        rem_clean = re.sub(r"^#\s*", "", rem).strip()
                        m_op = re.match(
                            r"^(?:===|==|==>|<=>|->|=>|➞|=|returns|should\s+return)\s*(?:#\s*)?(.+)$",
                            rem_clean,
                            flags=re.IGNORECASE,
                        )
                        if m_op:
                            e_lit = cls._clean_literal_expectation(m_op.group(1))
                            if e_lit is not None and not e_lit.startswith("="):
                                tests.append((c_expr, e_lit))
            i += 1

        seen = set()
        uniq = []
        for c_expr, e_expr in tests:
            try:
                ast.parse(c_expr, mode="eval")
                ast.literal_eval(e_expr)
                if (c_expr, e_expr) not in seen:
                    seen.add((c_expr, e_expr))
                    uniq.append((c_expr, e_expr))
            except Exception:
                continue
        return uniq[:10]

    @classmethod
    def synthesize_neural_spec_tests(
        cls, prompt_prefix: str, entry_point: str, generate_fn: Callable
    ) -> Tuple[List[Tuple[str, str]], int, float]:
        test_prompt = (
            f"Read the Python function docstring below and convert ONLY the explicit examples shown in the docstring into valid Python `assert {entry_point}(...) == expected_literal` lines.\n"
            f"Do NOT invent new test cases; translate the exact inputs and outputs given in the docstring text (including `Input:/Output:` blocks or alias calls) into calls to `{entry_point}`:\n"
            f"{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
            f"Return ONLY `assert {entry_point}(...) == ...` lines inside a single {FENCE}python ... {FENCE} block."
        )
        raw_asserts, tok, ms = generate_fn(test_prompt, max_new_tokens=220, temperature=0.0)
        synth_tests = []
        seen = set()
        for ln in raw_asserts.splitlines():
            ln_s = ln.strip()
            if "==" in ln_s and "(" in ln_s:
                clean_ln = re.sub(r"^assert\s+", "", ln_s)
                bal = cls._extract_balanced_call(clean_ln, entry_point)
                if bal:
                    c_expr, rem = bal
                    if rem.startswith("=="):
                        e_lit = cls._clean_literal_expectation(rem[2:])
                        if e_lit is not None:
                            try:
                                ast.parse(c_expr, mode="eval")
                                if (c_expr, e_lit) not in seen:
                                    seen.add((c_expr, e_lit))
                                    synth_tests.append((c_expr, e_lit))
                            except Exception:
                                pass
        return synth_tests[:8], tok, ms

    @classmethod
    def synthesize_system2_plan(
        cls, prompt_prefix: str, entry_point: str, generate_fn: Callable
    ) -> Tuple[str, int, float]:
        plan_prompt = (
            f"Analyze the Python function specification for `{entry_point}`:\n"
            f"{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
            f"Write a concise 3-bullet algorithmic specification checklist covering:\n"
            f"1. Exact input parsing / type handling (strings, negatives, floats, 0/1/primes).\n"
            f"2. Exact core formula, loop invariant, or string/list transformation required.\n"
            f"3. Efficiency guard to avoid infinite loops or timeouts (e.g., O(sqrt(n)) primality or greedy step)."
        )
        plan_txt, tok, ms = generate_fn(plan_prompt, max_new_tokens=180, temperature=0.0)
        return plan_txt.strip(), tok, ms

    @classmethod
    def synthesize_differential_fuzz_calls(cls, prompt_prefix: str, entry_point: str, num_probes: int = 24) -> List[str]:
        try:
            tree = cls._parse_prompt_ast(prompt_prefix)
            fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point)
            args = fn_node.args.args
            if not args:
                return [f"{entry_point}()"]

            rng = random.Random(42)
            pools = []
            for a in args:
                ann = ast.unparse(a.annotation).lower() if a.annotation else ""
                name = a.arg.lower()
                if "str" in ann or name in ("s", "string", "text", "word", "txt", "a", "b", "message"):
                    pools.append(["'a'", "'aba'", "'Hello world!'", "'5 apples and 6 oranges'", "'aBcDeF'", "'xyz'", "'I am bored. Are you?'", "'123'"])
                elif "list" in ann or "tuple" in ann or name in ("lst", "arr", "nums", "numbers", "grid", "l", "x", "grades"):
                    pools.append(["[1]", "[1, 2, 3]", "[-1, 0, 1]", "[3, 4, 5, 1, 2]", "[4.0, 3, 1.7, 2, 3.5]", "[2, 4, 6, 8]", "[1, 5, 2, 4, 3]"])
                elif "dict" in ann or name in ("d", "dct", "mapping"):
                    pools.append(["{}", "{'a': 1, 'b': 2}", "{'A': 1, 'b': 2}", "{1: 2}"])
                elif "float" in ann:
                    pools.append(["0.5", "1.5", "-2.5", "3.25", "10.0"])
                elif "bool" in ann:
                    pools.append(["True", "False"])
                else:
                    pools.append(["1", "2", "3", "4", "5", "6", "7", "10", "12", "19"])

            calls = []
            for idx in range(num_probes):
                chosen = [p[idx % len(p)] if idx < len(p) else rng.choice(p) for p in pools]
                calls.append(f"{entry_point}({', '.join(chosen)})")
            return list(dict.fromkeys(calls))
        except Exception:
            return []

    def trace_causal_slice(self, code_str: str, call_expr: str, entry_point: str) -> str:
        captured_locals = {}

        def tracer(frame, event, arg):
            if frame.f_code.co_name == entry_point and event in ("return", "exception", "line"):
                snap = {}
                for k, v in list(frame.f_locals.items())[:6]:
                    if not k.startswith("__") and not callable(v):
                        r = repr(v)
                        snap[k] = r[:60] + "..." if len(r) > 60 else r
                if snap:
                    captured_locals["state"] = snap
            return tracer

        env = build_safe_exec_env(self.stdlib_module)
        try:
            with _time_limit(0.20):
                exec(compile(code_str, "<candidate>", "exec"), env)
                sys.settrace(tracer)
                try:
                    eval(call_expr, env)
                except Exception:
                    pass
                finally:
                    sys.settrace(None)
        except Exception:
            sys.settrace(None)

        if "state" in captured_locals:
            return f"Causal Program Slice: {captured_locals['state']}"
        return "Causal Program Slice: <no local state captured>"

    def verify_tier1_and_tier2(
        self,
        code_str: str,
        prompt_prefix: str,
        entry_point: str,
        hard_doctests: Optional[List[Tuple[str, str]]] = None,
        soft_spec_tests: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        ok_g0, g0_msg = self.check_gate0_allowing_safe_stdlib(code_str)
        if not ok_g0:
            return {"passed": False, "score": -100, "soft_passed": 0, "doctest_count": 0, "tier": "GATE0_AST", "diagnostic": f"AST firewall: {g0_msg}"}

        env = build_safe_exec_env(self.stdlib_module)
        try:
            exec(compile(code_str, "<candidate>", "exec"), env)
        except Exception as exc:
            tb_last = traceback.format_exc().splitlines()[-1]
            return {"passed": False, "score": -50, "soft_passed": 0, "doctest_count": 0, "tier": "TIER1_COMPILE", "diagnostic": f"Syntax Error: {tb_last}"}

        if not callable(env.get(entry_point)):
            return {"passed": False, "score": -40, "soft_passed": 0, "doctest_count": 0, "tier": "TIER1_MISSING_FN", "diagnostic": f"Function `{entry_point}` missing"}

        doctests = hard_doctests if hard_doctests is not None else self.extract_docstring_tests(prompt_prefix, entry_point)
        passed_doctests = 0
        for call_expr, exp_expr in doctests:
            try:
                expected_val = ast.literal_eval(exp_expr)
            except Exception:
                continue

            try:
                with _time_limit(0.35):
                    actual_val = eval(call_expr, env)
            except Exception as exc:
                c_slice = self.trace_causal_slice(code_str, call_expr, entry_point)
                return {
                    "passed": False,
                    "score": passed_doctests * 20,
                    "soft_passed": 0,
                    "doctest_count": len(doctests),
                    "tier": "TIER1_DOCTEST_EXC",
                    "diagnostic": f"[DOCTEST_CRASH] `{call_expr}` raised {type(exc).__name__}: {exc}. Expected {repr(expected_val)}. {c_slice}",
                }

            matched = False
            if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)) and not isinstance(actual_val, bool) and not isinstance(expected_val, bool):
                matched = math.isclose(float(actual_val), float(expected_val), rel_tol=1e-5, abs_tol=1e-5)
            else:
                matched = (actual_val == expected_val)

            if not matched:
                c_slice = self.trace_causal_slice(code_str, call_expr, entry_point)
                return {
                    "passed": False,
                    "score": passed_doctests * 20,
                    "soft_passed": 0,
                    "doctest_count": len(doctests),
                    "tier": "TIER1_DOCTEST_MISMATCH",
                    "diagnostic": f"[DOCTEST_MISMATCH] On `{call_expr}`, expected {repr(expected_val)}, but returned {repr(actual_val)}. {c_slice}",
                }
            passed_doctests += 1

        soft_passed = 0
        soft_fail_diag = None
        if soft_spec_tests:
            for call_expr, exp_expr in soft_spec_tests:
                try:
                    expected_val = ast.literal_eval(exp_expr)
                except Exception:
                    continue
                try:
                    with _time_limit(0.35):
                        actual_val = eval(call_expr, env)
                except Exception as exc:
                    c_slice = self.trace_causal_slice(code_str, call_expr, entry_point)
                    return {
                        "passed": False,
                        "score": (passed_doctests * 20) + (soft_passed * 10),
                        "soft_passed": soft_passed,
                        "doctest_count": len(doctests),
                        "tier": "TIER1_PROSE_EXC",
                        "diagnostic": f"[DOCSTRING_EXAMPLE_CRASH] `{call_expr}` raised {type(exc).__name__}: {exc}. Expected {repr(expected_val)}. {c_slice}",
                    }

                matched_s = False
                if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)) and not isinstance(actual_val, bool) and not isinstance(expected_val, bool):
                    matched_s = math.isclose(float(actual_val), float(expected_val), rel_tol=1e-5, abs_tol=1e-5)
                else:
                    matched_s = (actual_val == expected_val)

                if matched_s:
                    soft_passed += 1
                elif soft_fail_diag is None:
                    c_slice = self.trace_causal_slice(code_str, call_expr, entry_point)
                    soft_fail_diag = f"[DOCSTRING_EXAMPLE_MISMATCH] On `{call_expr}`, expected {repr(expected_val)}, but returned {repr(actual_val)}. {c_slice}"

            if soft_fail_diag is not None and len(doctests) < 2:
                return {
                    "passed": False,
                    "score": (passed_doctests * 20) + (soft_passed * 10),
                    "soft_passed": soft_passed,
                    "doctest_count": len(doctests),
                    "tier": "TIER1_PROSE_MISMATCH",
                    "diagnostic": soft_fail_diag,
                }

        return {
            "passed": True,
            "score": 100 + (passed_doctests * 25) + (soft_passed * 12),
            "soft_passed": soft_passed,
            "soft_fail_diag": soft_fail_diag,
            "doctest_count": len(doctests),
            "tier": "PASS",
            "diagnostic": "PASS",
        }

    def rank_candidates_by_consensus(
        self,
        candidate_records: List[Dict[str, Any]],
        prompt_prefix: str,
        entry_point: str,
        hard_doctests: List[Tuple[str, str]],
        soft_spec_tests: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        if len(candidate_records) == 1:
            return candidate_records[0]

        fuzz_calls = (
            [c for c, _ in hard_doctests]
            + [c for c, _ in soft_spec_tests]
            + self.synthesize_differential_fuzz_calls(prompt_prefix, entry_point, num_probes=24)
        )
        fuzz_calls = list(dict.fromkeys(fuzz_calls))[:28]

        signatures = []
        valid_outputs_count = []
        for rec in candidate_records:
            c_code = rec["code"]
            env = build_safe_exec_env(self.stdlib_module)
            sig = []
            valid_cnt = 0
            try:
                exec(compile(c_code, "<candidate>", "exec"), env)
                for f_call in fuzz_calls:
                    try:
                        with _time_limit(0.15):
                            val = eval(f_call, env)
                        sig.append(repr(val))
                        valid_cnt += 1
                    except Exception as exc:
                        sig.append(f"EXC:{type(exc).__name__}")
            except Exception:
                sig.append("COMPILE_ERR")
            signatures.append(tuple(sig))
            valid_outputs_count.append(valid_cnt)

        cluster_counts = collections.Counter(signatures)
        best_rec = candidate_records[0]
        best_total_score = -1e9

        for idx, rec in enumerate(candidate_records):
            sig = signatures[idx]
            cluster_size = cluster_counts[sig] if "COMPILE_ERR" not in sig else 0
            greedy_bonus = 2.0 if rec.get("is_greedy") else 0.0
            total_score = (
                float(rec["ver"]["score"])
                + (cluster_size * 5.0)
                + (valid_outputs_count[idx] * 0.4)
                + greedy_bonus
            )
            if total_score > best_total_score:
                best_total_score = total_score
                best_rec = rec

        return best_rec

    @staticmethod
    def evaluate_canonical_vault(code_str: str, canonical_test_src: str, entry_point: str) -> Tuple[bool, str]:
        env = build_safe_exec_env()
        try:
            with _time_limit(0.60):
                exec(compile(code_str, "<candidate>", "exec"), env)
                exec(compile(canonical_test_src, "<vault_test>", "exec"), env)
                check_fn = env.get("check")
                cand_fn = env.get(entry_point)
                if not callable(check_fn) or not callable(cand_fn):
                    return False, "MissingEntrypoint"
                check_fn(cand_fn)
            return True, "PASS"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def synthesize_and_repair_task(
        self, task: Dict[str, Any], generate_fn: Callable, batch_generate_fn: Callable
    ) -> Dict[str, Any]:
        prompt_prefix = task["prompt"]
        entry_point = task["entry_point"]
        hard_doctests = self.extract_docstring_tests(prompt_prefix, entry_point)

        if len(hard_doctests) >= 2:
            fast_prompt = f"Write the Python function `{entry_point}` satisfying:\n{prompt_prefix}"
            raw_fast, tok_f, ms_f = generate_fn(fast_prompt, max_new_tokens=300, temperature=0.0)
            cand_fast = self.build_executable_module(prompt_prefix, raw_fast, entry_point)
            ver_fast = self.verify_tier1_and_tier2(cand_fast, prompt_prefix, entry_point, hard_doctests, None)
            if ver_fast["passed"]:
                return {
                    "passed_pre_vault": True,
                    "turns_used": 0,
                    "total_tokens": tok_f,
                    "total_latency_ms": round(ms_f, 2),
                    "final_code": cand_fast,
                }

        soft_spec_tests, st_tok, st_ms = self.synthesize_neural_spec_tests(prompt_prefix, entry_point, generate_fn)
        s2_plan, pl_tok, pl_ms = self.synthesize_system2_plan(prompt_prefix, entry_point, generate_fn)
        total_tokens = st_tok + pl_tok
        total_ms = st_ms + pl_ms

        p_raw = f"Write the Python function `{entry_point}` satisfying:\n{prompt_prefix}"
        p_plan = (
            f"Implement the Python function `{entry_point}` following this specification and algorithmic plan:\n"
            f"Specification:\n{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
            f"Algorithmic Checklist:\n{s2_plan}\n"
            f"Return ONLY valid Python code inside a single {FENCE}python ... {FENCE} block."
        )
        p_checklist = (
            f"Carefully read every rule in the docstring for `{entry_point}` (check 0/1/prime rules, negative digit signs, float/string parsing, tie-breaking, case sensitivity, and exact return types):\n"
            f"{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
            f"Write the complete, bug-free Python function `{entry_point}` inside a single {FENCE}python ... {FENCE} block."
        )
        p_pythonic = (
            f"Implement `{entry_point}` using clean, efficient Python standard library tools (`collections`, `itertools`, `math`, `re`, list comprehensions) so all docstring examples and time limits pass:\n"
            f"{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
            f"Return ONLY valid Python code inside a single {FENCE}python ... {FENCE} block."
        )

        raw_a, tok_a, ms_a = generate_fn(p_raw, max_new_tokens=300, temperature=0.0)
        raw_b, tok_b, ms_b = generate_fn(p_plan, max_new_tokens=350, temperature=0.0)
        raw_c, tok_c, ms_c = generate_fn(p_checklist, max_new_tokens=350, temperature=0.0)
        batch_b, tok_bb, ms_bb = batch_generate_fn(p_plan, max_new_tokens=350, temperature=0.35, num_samples=4)
        batch_c, tok_bc, ms_bc = batch_generate_fn(p_checklist, max_new_tokens=350, temperature=0.55, num_samples=4)
        batch_d, tok_bd, ms_bd = batch_generate_fn(p_pythonic, max_new_tokens=350, temperature=0.70, num_samples=4)

        total_tokens += (tok_a + tok_b + tok_c + tok_bb + tok_bc + tok_bd)
        total_ms += (ms_a + ms_b + ms_c + ms_bb + ms_bc + ms_bd)

        raw_pool = (
            [(raw_a, True), (raw_b, True), (raw_c, True)]
            + [(r, False) for r in batch_b]
            + [(r, False) for r in batch_c]
            + [(r, False) for r in batch_d]
        )

        all_history = []
        passing_pool = []

        for raw_str, is_greedy in raw_pool:
            c_mod = self.build_executable_module(prompt_prefix, raw_str, entry_point)
            v = self.verify_tier1_and_tier2(c_mod, prompt_prefix, entry_point, hard_doctests, soft_spec_tests)
            rec = {"code": c_mod, "ver": v, "is_greedy": is_greedy}
            all_history.append(rec)
            if v["passed"]:
                passing_pool.append(rec)

        if passing_pool:
            chosen = self.rank_candidates_by_consensus(
                passing_pool, prompt_prefix, entry_point, hard_doctests, soft_spec_tests
            )
            return {
                "passed_pre_vault": True,
                "turns_used": 1,
                "total_tokens": total_tokens,
                "total_latency_ms": round(total_ms, 2),
                "final_code": chosen["code"],
            }

        all_history.sort(key=lambda r: r["ver"]["score"], reverse=True)
        cand_code = all_history[0]["code"]
        seen_code_hashes = {hash(r["code"].strip()) for r in all_history}
        temp_schedule = [0.0, 0.25, 0.45, 0.65, 0.80]

        for turn in range(1, self.max_repair_turns + 1):
            ver = self.verify_tier1_and_tier2(cand_code, prompt_prefix, entry_point, hard_doctests, soft_spec_tests)
            if ver["passed"]:
                passing_pool.append({"code": cand_code, "ver": ver, "is_greedy": False})
                break

            repair_prompt = (
                f"Your previous Python implementation of `{entry_point}` failed verification:\n"
                f"{FENCE}python\n{cand_code.strip()}\n{FENCE}\n"
                f"Diagnostic & Causal Slice: {ver['diagnostic']}\n"
                f"Algorithmic Checklist:\n{s2_plan}\n"
                f"Original Specification:\n{FENCE}python\n{prompt_prefix.strip()}\n{FENCE}\n"
                f"Trace the failing input step by step, fix the bug, and return the complete corrected Python function `{entry_point}` inside a single {FENCE}python ... {FENCE} block."
            )
            temp = temp_schedule[min(turn - 1, len(temp_schedule) - 1)]
            rep_raw, r_tok, r_ms = generate_fn(repair_prompt, max_new_tokens=360, temperature=temp)
            total_tokens += r_tok
            total_ms += r_ms
            new_cand = self.build_executable_module(prompt_prefix, rep_raw, entry_point)
            new_ver = self.verify_tier1_and_tier2(new_cand, prompt_prefix, entry_point, hard_doctests, soft_spec_tests)
            all_history.append({"code": new_cand, "ver": new_ver, "is_greedy": (temp == 0.0)})
            if new_ver["passed"]:
                passing_pool.append({"code": new_cand, "ver": new_ver, "is_greedy": (temp == 0.0)})
                break

            if hash(new_cand.strip()) in seen_code_hashes or not new_ver["passed"]:
                rep_batch, rb_tok, rb_ms = batch_generate_fn(repair_prompt, max_new_tokens=360, temperature=0.55, num_samples=3)
                total_tokens += rb_tok
                total_ms += rb_ms
                for rb in rep_batch:
                    alt_cand = self.build_executable_module(prompt_prefix, rb, entry_point)
                    alt_ver = self.verify_tier1_and_tier2(alt_cand, prompt_prefix, entry_point, hard_doctests, soft_spec_tests)
                    rec_alt = {"code": alt_cand, "ver": alt_ver, "is_greedy": False}
                    all_history.append(rec_alt)
                    if alt_ver["passed"]:
                        passing_pool.append(rec_alt)
                        break
                    elif alt_ver["score"] >= new_ver["score"]:
                        new_cand = alt_cand
                        new_ver = alt_ver
                if passing_pool:
                    break

            seen_code_hashes.add(hash(new_cand.strip()))
            cand_code = new_cand

        if passing_pool:
            winner = self.rank_candidates_by_consensus(
                passing_pool, prompt_prefix, entry_point, hard_doctests, soft_spec_tests
            )
            return {
                "passed_pre_vault": True,
                "turns_used": self.max_repair_turns,
                "total_tokens": total_tokens,
                "total_latency_ms": round(total_ms, 2),
                "final_code": winner["code"],
            }

        viable = [r for r in all_history if r["ver"]["score"] >= 0] or all_history
        winner = self.rank_candidates_by_consensus(
            viable, prompt_prefix, entry_point, hard_doctests, soft_spec_tests
        )
        return {
            "passed_pre_vault": False,
            "turns_used": self.max_repair_turns,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_ms, 2),
            "final_code": winner["code"],
        }
