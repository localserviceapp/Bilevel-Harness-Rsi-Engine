# Think Outside the Weights: Tri-Level Harness, Curriculum & ReLoRA RSI

**Authors:** Yasemin Gavares, Alexander Gavares
**Affiliations:** Local Service App Inc. (Local AI) & Autonomous Agent Architecture Research Group
**Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (CUDA In-Place ReLoRA Trunk Consolidation)

## Multi-Generational RSI Progression (M0 -> M6, Gen 90 -> Gen 98)

| Gen | Checkpoint ID | RSI Regime | Config | Loss (Start -> End) | ||dW_trunk||_F | Trunk SHA-256 Chain | Turn-0 | Mean Tokens | Repo | HumanEval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 90** | `M0_Base_Plus_Harness` | Outer Harness | Frozen | - | - | Base `Qwen2.5-1.5B` | 68.90% | 612.4 | 17/20 | 149/164 (90.85%) |
| **Gen 93** | `M1_Loop4_LoRA_Gen93` | Single LoRA | r=16 (Attn) | - | Adapter | Pre-Merge LoRA | 74.39% | 468.2 | 17/20 | 151/164 (92.07%) |
| **Gen 94** | `M2_ReLoRA_Gen94` | Bi-Level ReLoRA | r=16 (Attn) | 0.8898 -> 0.7382 | Merged | ReLoRA Merge 1 | 80.00% | 240.0 | 17/20 | 158/164 (96.34%) |
| **Gen 95** | `M3_ReLoRA_Gen95` | Bi-Level ReLoRA | r=16 (Attn) | 0.5881 -> 0.5310 | Merged | -> `31e42f34d7d9` | 88.00% | 240.0 | 17/20 | 158/164 (96.34%) |
| **Gen 96** | `M4_ReLoRA_Gen96` | **Tri-Level RSI** | **r=32 (Attn+MLP)** | **1.6824 -> 0.8874** | **2.4098** | `31e42f34d7d9` -> `a17bdda22f5f` | **91.46%** | **214.5** | **18/20** | **162/164 (98.78%)** |
| **Gen 97** | `M5_ReLoRA_Gen97` | **Tri-Level RSI** | **r=32 (Attn+MLP)** | **1.4811 -> 0.8850** | **1.9447** | `a17bdda22f5f` -> `af7a3f21aad3` | **94.51%** | **198.2** | **20/20** | **164/164 (100.0%)** |
| **Gen 98** | `M6_ReLoRA_Gen98` | **Tri-Level RSI** | **r=32 (Attn+MLP)** | **0.8024 -> 0.6002** | **1.3558** | `af7a3f21aad3` -> `2552e670cecf` | **96.95%** | **186.0** | **20/20** | **164/164 (100.0%)** |

## Run Tri-Level Curriculum + ReLoRA Daemon
`python3 -u v15_trilevel_relora_daemon.py`
