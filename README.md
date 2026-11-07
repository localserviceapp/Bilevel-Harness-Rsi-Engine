# Think Outside the Weights: Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines

**Authors:** Yasemin Gavares, Alexander Gavares  
**Affiliations:** Autonomous Agent Architecture Research Group, Local Service App Inc. / Kronoz Technologies LLC  
**Paper (PDF):** [`Think_outside_the_weights_bi_level_harness_engine_evaluated.pdf`](./Think_outside_the_weights_bi_level_harness_engine_evaluated.pdf) (`v14.2.0` — 93-Generation Production Closure)

---

## Executive Overview

**Bilevel-Harness-Rsi-Engine (`v14.2.0`)** is a 4-Loop Recursive Self-Improvement (RSI) and formal verification engine that couples outer-loop cognitive harness search with inner-loop Low-Rank Adaptation (LoRA) weight distillation.

Using an unmodified **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** backbone (~3 GB FP16 VRAM) on a single CUDA GPU:
* **Canonical 164-Task OpenAI HumanEval (`openai/openai_humaneval`):** Lifts Pass@1 accuracy from **65.24% (107/164)** raw single-pass baseline to **90.85% (149/164)** via the outer-loop Bi-Level Harness (+25.61 percentage points, +42 net tasks rescued), and reaches **91.46% (150/164)** in a single pass (**92.07% [151/164]** cumulative, +44 net tasks rescued) after out-of-fold Loop-4 LoRA weight distillation.
* **Multi-File Repository Suite (`SWE-bench Verified` & `SWE-EVO`, 20 Tasks):** Lifts multi-file repository patching from **0.00% (0/20)** raw unified-diff baseline to **85.00% (17/20)** via cross-file AST fault localization and import-preserving `difflib` splicing (solving **10/10** on the `SWE-bench Verified` bug-resolution track and **7/10** on the `SWE-EVO` multi-file evolution track).

---

## Canonical 164-Task HumanEval Benchmark Comparison

*Note: Published reference scores for external models reflect standard single-pass evaluations reported in technical literature and typically vary slightly across evaluation harnesses and prompt templates.*

| System / Model | Parameters | Approx. FP16 VRAM | Evaluation Mode | HumanEval Solved | Pass@1 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen2.5-Coder-1.5B-Instruct` (Raw) | 1.5B | ~3 GB | Single-Pass Greedy ($T=0.0$) | 107 / 164 | 65.24% |
| `CodeLlama-70B-Instruct` | 70B | ~140 GB | Single-Pass Greedy | — | 67.8% |
| `Llama-3.1-70B-Instruct` | 70B | ~140 GB | Single-Pass Greedy | — | 80.5% |
| `Phi-4` | 14B | ~28 GB | Single-Pass Greedy | — | 82.6% |
| `Qwen2.5-Coder-7B-Instruct` | 7.6B | ~15 GB | Single-Pass Greedy | — | 84.8% |
| `Llama-3.3-70B-Instruct` | 70B | ~140 GB | Single-Pass Greedy | — | 88.4% |
| `Llama-3.1-405B-Instruct` | 405B | ~810 GB | Single-Pass Greedy | — | 89.0% |
| `Qwen2.5-Coder-14B-Instruct` | 14.7B | ~29 GB | Single-Pass Greedy | — | 89.6% |
| `GPT-4o (2024-05)` | Proprietary | Cloud API | Single-Pass 0-Shot | — | 90.2% |
| **`v14.0` Bi-Level Harness (Ours)** | **1.5B** | **~3 GB** | **Outer-Loop Bi-Level Cascade** | **149 / 164** | **90.85%** |
| **`v14.2` + Loop-4 LoRA RSI (Single-Run)** | **1.5B** | **~3 GB** | **4-Loop Parameter-Level RSI** | **150 / 164** | **91.46%** |
| **`v14.2` + Loop-4 LoRA RSI (Cumulative)** | **1.5B** | **~3 GB** | **4-Loop Parameter-Level RSI** | **151 / 164** | **92.07%** |
| `Claude 3.5 Sonnet` | Proprietary | Cloud API | Single-Pass 0-Shot | — | 92.0% |
| `Qwen2.5-Coder-32B-Instruct` | 32.5B | ~65 GB | Single-Pass Greedy | — | 92.7% |

---

## Generations 86–93 Ablation & Multi-File Repository Progression

| Benchmark / Stage | Raw 1.5B Baseline | Stage 1 AST Gate | Stage 2–3 Bi-Level Harness | Gen 93 Loop-4 LoRA RSI | Net Gain Over Raw 1.5B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HumanEval (164 Tasks, Single-Run)** | 107 / 164 (65.24%) | 133 / 164 (81.10%) | 149 / 164 (90.85%) | **150 / 164 (91.46%)** | **+26.22% (+43 tasks)** |
| **HumanEval (164 Tasks, Cumulative)** | 107 / 164 (65.24%) | 133 / 164 (81.10%) | 149 / 164 (90.85%) | **151 / 164 (92.07%)** | **+26.83% (+44 tasks)** |
| **HumanEval Tasks 1–70 Segment** | 57 / 70 (81.4%) | 67 / 70 (95.7%) | **70 / 70 (All Solved)** | **70 / 70 (All Solved)** | **+13 tasks** |
| **SWE-bench Verified Track (10 Tasks)** | 0 / 10 (0.0%) | 5 / 10 (50.0%) | **10 / 10 (All Solved)** | **10 / 10 (All Solved)** | **+10 repo tasks** |
| **SWE-EVO Evolution Track (10 Tasks)** | 0 / 10 (0.0%) | 5 / 10 (50.0%) | **7 / 10 (70.0%)** | **7 / 10 (70.0%)** | **+7 repo tasks** |
| **Combined Multi-File Repo Suite (20)** | 0 / 20 (0.0%) | 10 / 20 (50.0%) | **17 / 20 (85.0%)** | **17 / 20 (85.0%)** | **+85.0% (+17 tasks)** |

---

## Key Repository Artifacts

* `Think_outside_the_weights_bi_level_harness_engine_evaluated.pdf` — Full 93-generation technical report (`v14.2.0`).
* `harness/humaneval_full_sweep.py` — Unified Zero-Fallback Bi-Level Harness for canonical 164-task `openai_humaneval`.
* `harness/swebench_evo_sweep.py` — Cross-file AST fault localizer and import-preserving `difflib` patch stitcher for `SWE-bench Verified` and `SWE-EVO`.
* `run_v14_true_rsi_loop4.py` — Out-of-fold Loop-4 LoRA weight distillation (`r=8, alpha=16`) and unseen holdout evaluator.
* `v14_humaneval_164_metrics.json` & `v14_humaneval_checkpoint.jsonl` — Task-by-task telemetry across all 164 HumanEval problems.
* `v14_swebench_evo_metrics.json` & `v14_loop4_true_rsi_metrics.json` — Multi-file repository and Loop-4 LoRA distillation telemetry.
* `track2_map_elites_archive.db` — SQLite MAP-Elites quality-diversity archive across Generations 1–93.
