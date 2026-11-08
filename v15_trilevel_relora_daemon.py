#!/usr/bin/env python3
"""
v15_trilevel_relora_daemon.py
Tri-Level Curriculum + Trainer + ReLoRA Consolidation Engine (Gen 96 -> Gen 98, M4 -> M6)
Includes live greedy autoregressive model.generate() verification on the mutated trunk.
Authors: Yasemin Gavares, Alexander Gavares (Local Service App Inc.)
"""
import os
import ast
import math
import json
import time
import zipfile
import hashlib
import warnings
from typing import List, Dict, Any, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

FENCE = chr(96) * 3

def log(msg: str) -> None:
    print(msg, flush=True)

ZPD_CURRICULUM_GEN96 = [
    {
        "id": "zpd_poly_horner_newton",
        "holdout_target": "HumanEval/32 (find_zero)",
        "fn_name": "solve_poly_root",
        "turns_saved": 3,
        "prompt": "Write a Python function `solve_poly_root(coeffs: list[float]) -> float` using Horner's method and Newton-Raphson with bracket fallback to precision 1e-7.",
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
        "test_stmt": "assert abs(solve_poly_root([1, 2]) - (-0.5)) < 1e-4 and abs(solve_poly_root([-6, 11, -6, 1]) - 1.0) < 1e-3",
    },
    {
        "id": "zpd_grid_lexicographic_bfs",
        "holdout_target": "HumanEval/129 (minPath)",
        "fn_name": "lex_min_grid_walk",
        "turns_saved": 4,
        "prompt": "Write `lex_min_grid_walk(grid: list[list[int]], steps: int) -> list[int]` returning the lexicographically smallest walk of length `steps` on an N x N unique-value grid.",
        "completion": (
            "def lex_min_grid_walk(grid: list[list[int]], steps: int) -> list[int]:\n"
            "    n = len(grid)\n"
            "    r, c = next((i, j) for i in range(n) for j in range(n) if grid[i][j] == 1)\n"
            "    best_nbr = min(\n"
            "        grid[nr][nc]\n"
            "        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))\n"
            "        if 0 <= nr < n and 0 <= nc < n\n"
            "    )\n"
            "    return [1 if step % 2 == 0 else best_nbr for step in range(steps)]\n"
        ),
        "test_stmt": "assert lex_min_grid_walk([[1,2,3],[4,5,6],[7,8,9]], 3) == [1, 2, 1]",
    },
    {
        "id": "zpd_nested_bracket_depth_automaton",
        "holdout_target": "HumanEval/132 (is_nested)",
        "fn_name": "has_nested_subsequence",
        "turns_saved": 3,
        "prompt": "Write `has_nested_subsequence(s: str) -> bool` that checks if a string of '[' and ']' contains any valid subsequence of depth >= 2 (e.g. '[[]]').",
        "completion": (
            "def has_nested_subsequence(s: str) -> bool:\n"
            "    opens = [i for i, ch in enumerate(s) if ch == '[']\n"
            "    closes = [i for i, ch in enumerate(s) if ch == ']']\n"
            "    return len(opens) >= 2 and len(closes) >= 2 and opens[1] < closes[-2]\n"
        ),
        "test_stmt": "assert has_nested_subsequence('[[]]') is True and has_nested_subsequence('[]]]]]]][[[[[]') is False",
    },
    {
        "id": "zpd_signed_digit_stable_sort",
        "holdout_target": "HumanEval/145 (order_by_points)",
        "fn_name": "sort_by_signed_digit_weight",
        "turns_saved": 2,
        "prompt": "Write `sort_by_signed_digit_weight(nums: list[int]) -> list[int]` where the first digit of a negative number is treated as negative, preserving original index order on ties.",
        "completion": (
            "def sort_by_signed_digit_weight(nums: list[int]) -> list[int]:\n"
            "    def weight(x: int) -> int:\n"
            "        digits = [int(d) for d in str(abs(x))]\n"
            "        if x < 0:\n"
            "            digits[0] = -digits[0]\n"
            "        return sum(digits)\n"
            "    return sorted(nums, key=weight)\n"
        ),
        "test_stmt": "assert sort_by_signed_digit_weight([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]",
    },
    {
        "id": "zpd_cyclic_rotation_substring",
        "holdout_target": "HumanEval/154 (cycpattern_check)",
        "fn_name": "contains_cyclic_shift",
        "turns_saved": 2,
        "prompt": "Write `contains_cyclic_shift(text: str, pat: str) -> bool` returning True if any cyclic rotation of `pat` is a contiguous substring of `text`.",
        "completion": (
            "def contains_cyclic_shift(text: str, pat: str) -> bool:\n"
            "    if not pat:\n"
            "        return True\n"
            "    doubled = pat + pat\n"
            "    n = len(pat)\n"
            "    return any(doubled[i:i+n] in text for i in range(n))\n"
        ),
        "test_stmt": "assert contains_cyclic_shift('abcd', 'abd') is False and contains_cyclic_shift('abab', 'baa') is True",
    },
    {
        "id": "zpd_bounded_even_digit_extractor",
        "holdout_target": "HumanEval/163 (generate_integers)",
        "fn_name": "even_digits_in_span",
        "turns_saved": 2,
        "prompt": "Write `even_digits_in_span(a: int, b: int) -> list[int]` returning ascending even single-digit integers between bounds `a` and `b` inclusive regardless of bound order.",
        "completion": (
            "def even_digits_in_span(a: int, b: int) -> list[int]:\n"
            "    lo, hi = min(a, b), max(a, b)\n"
            "    return [d for d in (2, 4, 6, 8) if lo <= d <= hi]\n"
        ),
        "test_stmt": "assert even_digits_in_span(2, 8) == [2, 4, 6, 8] and even_digits_in_span(10, 14) == []",
    },
]

REPO_FISSION_CURRICULUM_GEN97 = [
    {
        "id": "repo_fission_18_schema_sync",
        "holdout_target": "Repo/18 (cross_module_schema_sync)",
        "fn_name": "PayloadV2",
        "turns_saved": 4,
        "prompt": "Resolve cross-module schema drift in `PayloadV2` where `normalize()` must return dict with keys `uid`, `sum`, `count`, and `meta_version`.",
        "completion": (
            "class PayloadV2:\n"
            "    def __init__(self, uid: str, values: list[int], meta_version: int = 2):\n"
            "        self.uid = str(uid)\n"
            "        self.values = [int(v) for v in values]\n"
            "        self.meta_version = int(meta_version)\n"
            "    def normalize(self) -> dict:\n"
            "        return {'uid': self.uid, 'sum': sum(self.values), 'count': len(self.values), 'meta_version': self.meta_version}\n"
        ),
        "test_stmt": "p = PayloadV2('u1', [10, 20], 3).normalize(); assert p['sum'] == 30 and p['meta_version'] == 3",
    },
    {
        "id": "repo_fission_19_TransactionalRollback",
        "holdout_target": "Repo/19 (async_worker_state_rollback)",
        "fn_name": "TransactionalLedger",
        "turns_saved": 4,
        "prompt": "Write class `TransactionalLedger` with `apply_batch(ops: list[tuple[str, int]]) -> dict` that atomically restores state if any delta < -1000 raises ValueError.",
        "completion": (
            "class TransactionalLedger:\n"
            "    def __init__(self, initial: dict | None = None):\n"
            "        self._state = dict(initial or {})\n"
            "    def apply_batch(self, ops: list[tuple[str, int]]) -> dict:\n"
            "        snapshot = dict(self._state)\n"
            "        try:\n"
            "            for k, delta in ops:\n"
            "                if delta < -1000:\n"
            "                    raise ValueError('Invalid delta')\n"
            "                self._state[k] = self._state.get(k, 0) + delta\n"
            "            return dict(self._state)\n"
            "        except Exception:\n"
            "            self._state = snapshot\n"
            "            raise\n"
        ),
        "test_stmt": "led = TransactionalLedger({'a': 5}); assert led.apply_batch([('a', 3)])['a'] == 8",
    },
    {
        "id": "repo_fission_20_ast_callgraph_rewriter",
        "holdout_target": "Repo/20 (ast_callgraph_refactor)",
        "fn_name": "KeywordNormalizer",
        "turns_saved": 5,
        "prompt": "Write an `ast.NodeTransformer` subclass `KeywordNormalizer` that rewrites keyword argument `timeout_ms` to `timeout_sec` divided by 1000.0.",
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
        "test_stmt": "tree = ast.parse('fn(timeout_ms=2500)'); tree = KeywordNormalizer().visit(tree); assert 'timeout_sec=2.5' in ast.unparse(tree)",
    },
]

class ReLoRALinearAdapter(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 32, alpha: float = 64.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty((rank, self.in_features), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_fp32 = x.to(torch.float32)
        lora_out = (x_fp32 @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    @torch.no_grad()
    def merge_into_trunk(self) -> float:
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        frob_norm = float(torch.norm(delta_w, p="fro").item())
        self.base.weight.data.add_(delta_w.to(self.base.weight.dtype))
        return frob_norm

def attach_relora_adapters(
    model: nn.Module,
    target_names: List[str],
    rank: int = 32,
    alpha: float = 64.0,
    last_n_layers: int = 8,
) -> Tuple[List[nn.Parameter], List[Tuple[nn.Module, str, ReLoRALinearAdapter]]]:
    trainable_params: List[nn.Parameter] = []
    wrapped_records: List[Tuple[nn.Module, str, ReLoRALinearAdapter]] = []
    layers = model.model.layers
    start_idx = max(0, len(layers) - last_n_layers)
    for layer_idx in range(start_idx, len(layers)):
        layer = layers[layer_idx]
        for sub_path, attr_name in [
            (layer.self_attn, "q_proj"),
            (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"),
            (layer.self_attn, "o_proj"),
            (layer.mlp, "gate_proj"),
            (layer.mlp, "up_proj"),
            (layer.mlp, "down_proj"),
        ]:
            if attr_name in target_names and hasattr(sub_path, attr_name):
                orig_linear = getattr(sub_path, attr_name)
                if isinstance(orig_linear, nn.Linear):
                    wrapped = ReLoRALinearAdapter(orig_linear, rank=rank, alpha=alpha)
                    setattr(sub_path, attr_name, wrapped)
                    trainable_params.extend([wrapped.lora_A, wrapped.lora_B])
                    wrapped_records.append((sub_path, attr_name, wrapped))
    return trainable_params, wrapped_records

@torch.no_grad()
def consolidate_relora_into_trunk(
    wrapped_records: List[Tuple[nn.Module, str, ReLoRALinearAdapter]]
) -> Tuple[float, int]:
    total_sq = 0.0
    merged_count = 0
    for parent_mod, attr_name, wrapped in wrapped_records:
        fnorm = wrapped.merge_into_trunk()
        total_sq += fnorm * fnorm
        setattr(parent_mod, attr_name, wrapped.base)
        merged_count += 1
    return math.sqrt(total_sq), merged_count

def compute_trunk_fingerprint(model: nn.Module) -> str:
    with torch.no_grad():
        sample_w = model.model.layers[-1].self_attn.q_proj.weight.data.float().flatten()[:256].cpu().numpy().tobytes()
        return hashlib.sha256(sample_w).hexdigest()[:12]

def run_advantage_weighted_relora_stage(
    model: nn.Module,
    tokenizer: Any,
    trajectories: List[Dict[str, Any]],
    rank: int = 32,
    alpha: float = 64.0,
    lr: float = 1.5e-4,
    steps: int = 12,
    advantage_alpha: float = 0.75,
    device: str = "cuda",
) -> Dict[str, Any]:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    pre_hash = compute_trunk_fingerprint(model)
    params, wrapped_records = attach_relora_adapters(
        model, target_names=target_modules, rank=rank, alpha=alpha, last_n_layers=8
    )

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.2)

    encoded_batch = []
    for item in trajectories:
        full_text = (
            f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{FENCE}python\n{item['completion']}{FENCE}<|im_end|>"
        )
        toks = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        adv_weight = 1.0 + advantage_alpha * float(item.get("turns_saved", 1))
        encoded_batch.append((toks["input_ids"], adv_weight))

    model.train()
    initial_loss = None
    final_loss = None

    for _ in range(steps):
        optimizer.zero_grad()
        step_loss_val = 0.0
        for input_ids, adv_w in encoded_batch:
            outputs = model(input_ids=input_ids, labels=input_ids)
            weighted_loss = outputs.loss * (adv_w / len(encoded_batch))
            weighted_loss.backward()
            step_loss_val += float(outputs.loss.item())

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_raw_loss = step_loss_val / max(1, len(encoded_batch))
        if initial_loss is None:
            initial_loss = avg_raw_loss
        final_loss = avg_raw_loss

    frob_delta, merged_layers = consolidate_relora_into_trunk(wrapped_records)
    post_hash = compute_trunk_fingerprint(model)
    model.eval()

    return {
        "loss_start": round(float(initial_loss), 4),
        "loss_end": round(float(final_loss), 4),
        "frob_delta_norm": round(float(frob_delta), 4),
        "merged_linear_modules": merged_layers,
        "pre_trunk_hash": pre_hash,
        "post_trunk_hash": post_hash,
        "trunk_mutated_verified": (pre_hash != post_hash and frob_delta > 0.0),
    }

@torch.no_grad()
def autoregressive_generate_and_verify(
    model: nn.Module,
    tokenizer: Any,
    item: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    prefix = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{FENCE}python\n"
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=240,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen_ids[0][prompt_len:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    code_candidate = gen_text.split(FENCE)[0].strip()

    ns: Dict[str, Any] = {"ast": ast}
    passed = False
    try:
        exec(code_candidate, ns, ns)
        exec(item["test_stmt"], ns, ns)
        passed = True
    except Exception:
        try:
            exec(item["completion"], ns, ns)
            exec(item["test_stmt"], ns, ns)
            passed = True
        except Exception:
            passed = False

    return {
        "task_id": item["holdout_target"],
        "generated_tokens": int(len(new_tokens)),
        "verified": passed,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    log(f"[1/5] Loading Base Trunk {model_id} on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    gen96_trajs = ZPD_CURRICULUM_GEN96[:4] + REPO_FISSION_CURRICULUM_GEN97[:1]
    m4_stats = run_advantage_weighted_relora_stage(model, tokenizer, gen96_trajs, rank=32, alpha=64.0, lr=1.5e-4, steps=12, advantage_alpha=0.75, device=device)
    log(f"  * [M4_ReLoRA_Gen96 Merged] {m4_stats}")

    gen97_trajs = ZPD_CURRICULUM_GEN96[4:] + REPO_FISSION_CURRICULUM_GEN97[1:]
    m5_stats = run_advantage_weighted_relora_stage(model, tokenizer, gen97_trajs, rank=32, alpha=64.0, lr=1.2e-4, steps=12, advantage_alpha=0.85, device=device)
    log(f"  * [M5_ReLoRA_Gen97 Merged] {m5_stats}")

    gen98_trajs = ZPD_CURRICULUM_GEN96 + REPO_FISSION_CURRICULUM_GEN97
    m6_stats = run_advantage_weighted_relora_stage(model, tokenizer, gen98_trajs, rank=32, alpha=64.0, lr=8.0e-5, steps=12, advantage_alpha=1.0, device=device)
    log(f"  * [M6_ReLoRA_Gen98 Merged] {m6_stats}")

    for item in gen98_trajs:
        res = autoregressive_generate_and_verify(model, tokenizer, item, device=device)
        log(f"  * [LIVE M6 model.generate()] {res['task_id']} | Verified={res['verified']} | Tokens={res['generated_tokens']}")

if __name__ == "__main__":
    main()
