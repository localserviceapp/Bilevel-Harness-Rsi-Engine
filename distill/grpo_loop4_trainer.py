import ast
import copy
import json
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

FORBIDDEN_IDENTIFIERS = {"eval", "exec", "__import__", "tb_frame", "gi_frame", "globals", "locals"}

def extract_signature(code_str: str) -> str:
    for line in code_str.splitlines():
        if line.strip().startswith("def "):
            return line.strip()
    return "def solution(x):"

def check_gate0_ast(code_str: str) -> bool:
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
            return False
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
            return False
    return True

def extract_code_block(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def load_trajectory_dataset(jsonl_path: str = "verified_trajectories_30gen.jsonl") -> Dataset:
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("vault_pass_rate", 0.0) < 1.0:
                continue
            sig = extract_signature(r["verified_system1_code"])
            s2_text = "\n".join(r.get("system2_trace", []))
            prompt = (
                f"### Domain Archetype: {r['domain_archetype']}\n"
                f"### Problem ID: {r['problem_id']}\n"
                f"### Specification:\n{r['specification']}\n\n"
                f"Emit ONLY the verified Python implementation using exact signature `{sig}`:\n"
            )
            records.append({
                "prompt": prompt,
                "verified_system1_code": r["verified_system1_code"],
                "system2_len": max(len(s2_text), 1),
                "domain_archetype": r["domain_archetype"],
            })
    return Dataset.from_list(records)

def generate_test_inputs(domain: str):
    if domain == "graph_routing":
        return [
            [[999999, 4, 2, 999999], [999999, 999999, 5, 10], [999999, 999999, 999999, 3], [999999]*4],
            [[999999, 1, 999999], [999999, 999999, 2], [999999, 999999, 999999]],
        ]
    return [10, 25, [1, 2, 3, 4, 5]]

def loop4_rlvr_reward(completions, verified_system1_code, system2_len, domain_archetype, **kwargs):
    rewards = []
    for comp, ref_code, s2_len, dom in zip(completions, verified_system1_code, system2_len, domain_archetype):
        raw = comp[0]["content"] if isinstance(comp, list) else str(comp)
        cand_code = extract_code_block(raw)

        # Gate 0: Static AST & zero-reflection enforcement
        if not check_gate0_ast(cand_code):
            rewards.append(-1.0)
            continue

        try:
            ref_ns, cand_ns = {}, {}
            exec(compile(ref_code, "<ref>", "exec"), {"__builtins__": __builtins__}, ref_ns)
            exec(compile(cand_code, "<cand>", "exec"), {"__builtins__": __builtins__}, cand_ns)

            fn_name = extract_signature(ref_code).split("def ")[1].split("(")[0].strip()
            if fn_name not in cand_ns or not callable(cand_ns[fn_name]):
                rewards.append(-0.5)
                continue

            # Gate 2: Deep-copy state isolation functional check across test inputs
            passed_all = True
            for test_in in generate_test_inputs(dom):
                expected = ref_ns[fn_name](copy.deepcopy(test_in))
                actual = cand_ns[fn_name](copy.deepcopy(test_in))
                if actual != expected:
                    passed_all = False
                    break

            # Information-theoretic policy compression bonus (\rho_\tau)
            rho_tau = s2_len / max(len(cand_code), 1)
            compression_bonus = min(0.5, 0.1 * rho_tau)
            rewards.append((2.0 + compression_bonus) if passed_all else 0.25)
        except Exception:
            rewards.append(-0.5)
    return rewards

def main():
    dataset = load_trajectory_dataset()
    print(f"Loaded {len(dataset)} verified trajectories (vault_pass_rate = 1.0).")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir="./lora_loop4_adapters",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=3,
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        reward_funcs=loop4_rlvr_reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model("./lora_loop4_adapters/system1_distilled")
    print("Saved Loop 4 System-1 LoRA adapter to ./lora_loop4_adapters/system1_distilled")

if __name__ == "__main__":
    main()
