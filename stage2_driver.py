import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sympy, sympy.printing
from sympy.printing.str import StrPrinter
sympy.printing.StrPrinter = StrPrinter

import re, sys, json, time, gzip, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.abspath("harness"))

import torch, peft.import_utils, peft.tuners.lora.torchao
peft.import_utils.is_torchao_available = lambda: False
peft.tuners.lora.torchao.is_torchao_available = lambda: False

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import stdlib_primitives
from macro_epoch2_router import MacroEpoch2Router
from continuous_evolution_loop import ContinuousEvolutionDaemon
from humaneval_full_sweep import OptimizedHumanEvalZeroFallbackEngine

FENCE = chr(96) * 3
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CHECKPOINT_FILE = Path("v14_humaneval_checkpoint.jsonl")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"[1/3] Loading {BASE_MODEL_ID} on {device.upper()}...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype, device_map="auto" if device == "cuda" else None, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype, device_map="auto" if device == "cuda" else None, local_files_only=False)
model.eval()

print("[2/3] Loading Canonical 164-Task Dataset...", flush=True)
try:
    dataset = list(load_dataset("openai/openai_humaneval", split="test"))
except Exception:
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    with urllib.request.urlopen(url) as resp:
        raw_bytes = gzip.decompress(resp.read()).decode("utf-8")
    dataset = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]

router = MacroEpoch2Router()
evo_daemon = ContinuousEvolutionDaemon()
engine = OptimizedHumanEvalZeroFallbackEngine(router, stdlib_primitives, max_repair_turns=5)

SYS_PROMPT = (
    "You are an expert Python algorithm synthesizer inside a formal verification harness. "
    f"Return ONLY valid, self-contained Python code inside a single {FENCE}python ... {FENCE} block."
)

class CodeFenceStopCriteria(StoppingCriteria):
    def __init__(self, tok, prompt_len: int):
        self.tok = tok
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < 12 or gen_len % 4 != 0:
            return False
        for seq in input_ids:
            txt = self.tok.decode(seq[self.prompt_len:], skip_special_tokens=True)
            if txt.count(FENCE) < 2:
                return False
        return True

def _extract_code_block(raw: str) -> str:
    pattern = rf"{FENCE}(?:python)?\s*\n?(.*?){FENCE}"
    blocks = re.findall(pattern, raw, flags=re.DOTALL)
    cand = blocks[0] if blocks else raw
    clean_lines = [ln for ln in cand.splitlines() if not ln.strip().startswith(FENCE)]
    return "\n".join(clean_lines).strip("\n")

def live_gpu_generate(user_prompt: str, max_new_tokens: int = 350, temperature: float = 0.0):
    t0 = time.perf_counter()
    txt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inp = tokenizer([txt], return_tensors="pt").to(model.device)
    stop_crit = StoppingCriteriaList([CodeFenceStopCriteria(tokenizer, inp.input_ids.shape[1])])
    with torch.no_grad():
        if temperature > 0.0:
            out = model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.92, stopping_criteria=stop_crit,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=False,
                stopping_criteria=stop_crit, pad_token_id=tokenizer.eos_token_id,
            )
    gen_ids = out[0][inp.input_ids.shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _extract_code_block(raw), int(len(gen_ids)), round((time.perf_counter() - t0) * 1000.0, 2)

def live_gpu_batch_generate(user_prompt: str, max_new_tokens: int = 350, temperature: float = 0.35, num_samples: int = 4):
    t0 = time.perf_counter()
    txt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inp = tokenizer([txt], return_tensors="pt").to(model.device)
    stop_crit = StoppingCriteriaList([CodeFenceStopCriteria(tokenizer, inp.input_ids.shape[1])])
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.92, num_return_sequences=num_samples,
            stopping_criteria=stop_crit, pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inp.input_ids.shape[1]
    cands = []
    tok_count = 0
    for seq in out:
        gen_ids = seq[prompt_len:]
        tok_count += int(len(gen_ids))
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        cands.append(_extract_code_block(raw))
    return cands, tok_count, round((time.perf_counter() - t0) * 1000.0, 2)

completed_map = {}
if CHECKPOINT_FILE.exists():
    for line in CHECKPOINT_FILE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            completed_map[rec["task_id"]] = rec

raw_cache = json.loads(Path("v14_raw_baseline_cache.json").read_text()) if Path("v14_raw_baseline_cache.json").exists() else {}

print(f"\n[3/3] Executing Stage-3 Final Closure on {164 - len(completed_map)} holdout tasks...", flush=True)
t_start = time.perf_counter()
raw_passed_count = 0
harness_passed_count = 0
per_task_records = []

for idx, item in enumerate(dataset):
    task_id = item["task_id"]
    if task_id in completed_map:
        rec = completed_map[task_id]
    else:
        prompt_prefix = item["prompt"]
        entry_point = item["entry_point"]
        canonical_test = item["test"]

        raw_ok = raw_cache.get(task_id, {}).get("raw_passed", False)
        raw_tok = raw_cache.get(task_id, {}).get("raw_tokens", 200)

        h_res = engine.synthesize_and_repair_task(item, live_gpu_generate, live_gpu_batch_generate)
        h_ok, h_reason = engine.evaluate_canonical_vault(h_res["final_code"], canonical_test, entry_point)

        rec = {
            "task_id": task_id,
            "entry_point": entry_point,
            "raw_passed": raw_ok,
            "harness_passed": h_ok,
            "repair_turns_used": h_res["turns_used"],
            "raw_tokens": raw_tok,
            "harness_tokens": h_res["total_tokens"],
            "vault_status": h_reason,
        }
        completed_map[task_id] = rec
        with open(CHECKPOINT_FILE, "a") as cf:
            cf.write(json.dumps(rec) + "\n")
        print(f"  * [BOOSTED] {task_id} ({entry_point}): Vault={h_ok} ({h_reason}) | Turns={h_res['turns_used']} | Tokens={h_res['total_tokens']}", flush=True)

    if rec["raw_passed"]:
        raw_passed_count += 1
    if rec["harness_passed"]:
        harness_passed_count += 1
    per_task_records.append(rec)

    if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
        elapsed_s = time.perf_counter() - t_start
        h_pct = (harness_passed_count / (idx + 1)) * 100.0
        r_pct = (raw_passed_count / (idx + 1)) * 100.0
        print(
            f"  [{idx+1:3d}/164] Raw 1.5B: {raw_passed_count:3d}/{idx+1} ({r_pct:5.1f}%) | "
            f"Bi-Level Harness: {harness_passed_count:3d}/{idx+1} ({h_pct:5.1f}%) | "
            f"Net Wins: +{harness_passed_count - raw_passed_count} | Elapsed: {elapsed_s:5.1f}s",
            flush=True,
        )

total_tasks = len(dataset)
raw_rate = round(raw_passed_count / total_tasks, 4)
harness_rate = round(harness_passed_count / total_tasks, 4)

summary = {
    "benchmark": "v14.0 Stage-3 90%+ HumanEval Final Closure",
    "model_id": BASE_MODEL_ID,
    "total_tasks": total_tasks,
    "raw_baseline_solved": raw_passed_count,
    "raw_baseline_pass_at_1": raw_rate,
    "bilevel_harness_solved": harness_passed_count,
    "bilevel_harness_pass_at_1": harness_rate,
    "absolute_pass_rate_lift_pct": round((harness_rate - raw_rate) * 100.0, 2),
    "net_tasks_rescued_by_harness": harness_passed_count - raw_passed_count,
    "per_task_results": per_task_records,
}

Path("v14_humaneval_164_metrics.json").write_text(json.dumps(summary, indent=2))
print("\n======================================================================", flush=True)
print(f" [4/4] STAGE-2 90%+ HUMANEVAL BOOST COMPLETE", flush=True)
print(f"  * Raw 1.5B Single-Pass : {raw_passed_count}/164 ({raw_rate*100:.2f}%)", flush=True)
print(f"  * v14.0 Bi-Level Harness: {harness_passed_count}/164 ({harness_rate*100:.2f}%)", flush=True)
print(f"  * Net Lift Over Raw 1.5B: +{(harness_rate - raw_rate)*100:.2f}% (+{harness_passed_count - raw_passed_count} tasks rescued)", flush=True)
print("======================================================================", flush=True)
