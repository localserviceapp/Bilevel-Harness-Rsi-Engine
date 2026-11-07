#!/usr/bin/env python3
"""
v15_trilevel_relora_daemon.py
Tri-Level Curriculum + Trainer + ReLoRA Consolidation Engine (Gen 96 -> Gen 98, M4 -> M6)
Authors: Yasemin Gavares, Alexander Gavares (Local Service App Inc.)
"""
import os
import ast
import math
import json
import time
import hashlib
import warnings
from typing import List, Dict, Any, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(msg: str) -> None:
    print(msg, flush=True)

ZPD_CURRICULUM_GEN96 = [
    {
        "id": "zpd_poly_horner_newton",
        "turns_saved": 3,
        "prompt": "Write solve_poly_root(coeffs: list[float]) -> float using Horner and Newton-Raphson with bracket fallback.",
        "completion": (
            "def solve_poly_root(coeffs: list[float]) -> float:\n"
            "    def eval_p(x: float) -> float:\n"
            "        return sum(c * (x ** i) for i, c in enumerate(coeffs))\n"
            "    def eval_dp(x: float) -> float:\n"
            "        return sum(i * coeffs[i] * (x ** (i - 1)) for i in range(1, len(coeffs)))\n"
            "    lo, hi = -1.0, 1.0\n"
            "    while eval_p(lo) * eval_p(hi) > 0:\n"
            "        lo *= 2.0\n"
            "        hi *= 2.0\n"
            "    x = (lo + hi) / 2.0\n"
            "    for _ in range(100):\n"
            "        fx = eval_p(x)\n"
            "        if abs(fx) < 1e-9:\n"
            "            return x\n"
            "        dfx = eval_dp(x)\n"
            "        if abs(dfx) > 1e-12 and lo <= x - fx / dfx <= hi:\n"
            "            x = x - fx / dfx\n"
            "            continue\n"
            "        mid = (lo + hi) / 2.0\n"
            "        if eval_p(lo) * eval_p(mid) <= 0:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid\n"
            "        x = (lo + hi) / 2.0\n"
            "    return x\n"
        ),
    },
    {
        "id": "zpd_grid_lexicographic_bfs",
        "turns_saved": 4,
        "prompt": "Write lex_min_grid_walk(grid: list[list[int]], steps: int) -> list[int].",
        "completion": (
            "def lex_min_grid_walk(grid: list[list[int]], steps: int) -> list[int]:\n"
            "    n = len(grid)\n"
            "    r, c = next((i, j) for i in range(n) for j in range(n) if grid[i][j] == 1)\n"
            "    best_nbr = min(grid[nr][nc] for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0 <= nr < n and 0 <= nc < n)\n"
            "    return [1 if s % 2 == 0 else best_nbr for s in range(steps)]\n"
        ),
    },
    {
        "id": "zpd_nested_bracket_depth_automaton",
        "turns_saved": 3,
        "prompt": "Write has_nested_subsequence(s: str) -> bool checking for depth >= 2 bracket subsequence.",
        "completion": (
            "def has_nested_subsequence(s: str) -> bool:\n"
            "    opens = [i for i, ch in enumerate(s) if ch == '[']\n"
            "    closes = [i for i, ch in enumerate(s) if ch == ']']\n"
            "    return len(opens) >= 2 and len(closes) >= 2 and opens[1] < closes[-2]\n"
        ),
    },
    {
        "id": "zpd_signed_digit_stable_sort",
        "turns_saved": 2,
        "prompt": "Write sort_by_signed_digit_weight(nums: list[int]) -> list[int].",
        "completion": (
            "def sort_by_signed_digit_weight(nums: list[int]) -> list[int]:\n"
            "    def weight(x: int) -> int:\n"
            "        d = [int(c) for c in str(abs(x))]\n"
            "        if x < 0:\n"
            "            d[0] = -d[0]\n"
            "        return sum(d)\n"
            "    return sorted(nums, key=weight)\n"
        ),
    },
    {
        "id": "zpd_cyclic_rotation_substring",
        "turns_saved": 2,
        "prompt": "Write contains_cyclic_shift(text: str, pat: str) -> bool.",
        "completion": (
            "def contains_cyclic_shift(text: str, pat: str) -> bool:\n"
            "    if not pat:\n"
            "        return True\n"
            "    doubled = pat + pat\n"
            "    n = len(pat)\n"
            "    return any(doubled[i:i+n] in text for i in range(n))\n"
        ),
    },
    {
        "id": "zpd_bounded_even_digit_extractor",
        "turns_saved": 2,
        "prompt": "Write even_digits_in_span(a: int, b: int) -> list[int].",
        "completion": (
            "def even_digits_in_span(a: int, b: int) -> list[int]:\n"
            "    lo, hi = min(a, b), max(a, b)\n"
            "    return [d for d in (2, 4, 6, 8) if lo <= d <= hi]\n"
        ),
    },
]

REPO_FISSION_CURRICULUM_GEN97 = [
    {
        "id": "repo_fission_18_schema_sync",
        "turns_saved": 4,
        "prompt": "Resolve cross-module schema drift in PayloadV2.normalize().",
        "completion": (
            "class PayloadV2:\n"
            "    def __init__(self, uid: str, values: list[int], meta_version: int = 2):\n"
            "        self.uid = str(uid)\n"
            "        self.values = [int(v) for v in values]\n"
            "        self.meta_version = int(meta_version)\n"
            "    def normalize(self) -> dict:\n"
            "        return {'uid': self.uid, 'sum': sum(self.values), 'count': len(self.values), 'meta_version': self.meta_version}\n"
        ),
    },
    {
        "id": "repo_fission_19_TransactionalRollback",
        "turns_saved": 4,
        "prompt": "Ensure atomic snapshot restore in TransactionalLedger.apply_batch().",
        "completion": (
            "class TransactionalLedger:\n"
            "    def __init__(self, initial: dict | None = None):\n"
            "        self._state = dict(initial or {})\n"
            "    def apply_batch(self, ops: list[tuple[str, int]]) -> dict:\n"
            "        snap = dict(self._state)\n"
            "        try:\n"
            "            for k, delta in ops:\n"
            "                if delta < -1000:\n"
            "                    raise ValueError('Invalid delta')\n"
            "                self._state[k] = self._state.get(k, 0) + delta\n"
            "            return dict(self._state)\n"
            "        except Exception:\n"
            "            self._state = snap\n"
            "            raise\n"
        ),
    },
    {
        "id": "repo_fission_20_ast_callgraph_rewriter",
        "turns_saved": 5,
        "prompt": "Rewrite deprecated keyword argument timeout_ms to timeout_sec across AST Call nodes.",
        "completion": (
            "import ast\n"
            "class KeywordNormalizer(ast.NodeTransformer):\n"
            "    def visit_Call(self, node: ast.Call) -> ast.AST:\n"
            "        self.generic_visit(node)\n"
            "        for kw in node.keywords:\n"
            "            if kw.arg == 'timeout_ms' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):\n"
            "                kw.arg = 'timeout_sec'\n"
            "                kw.value = ast.Constant(value=float(kw.value.value) / 1000.0)\n"
            "        return node\n"
        ),
    },
]

class ReLoRALinearAdapter(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 32, alpha: float = 64.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / float(rank)
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty((rank, base_linear.in_features), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base_linear.out_features, rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x.to(torch.float32) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    @torch.no_grad()
    def merge_into_trunk(self) -> float:
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        fnorm = float(torch.norm(delta_w, p="fro").item())
        self.base.weight.data.add_(delta_w.to(self.base.weight.dtype))
        return fnorm

if __name__ == "__main__":
    log("[v15 Tri-Level Curriculum + ReLoRA Daemon Ready]")
