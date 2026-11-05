# Bi-Level Harness Optimization & Textual Backpropagation in Autonomous 4-Loop RSI Engines

**Authors:** Yasemin Gavares & Alexander Gavares  
**Affiliations:** Autonomous Agent Architecture Research Group | Local Service App Inc (Local AI) / Kronoz Technologies LLC  
**Engine Version:** `4.0.0-open-ended-discovery` | **License:** CC BY 4.0

---

## Overview

Official reference implementation and empirical verification artifacts for **Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement (RSI) Engines**.

Our architecture decouples fast non-parametric cognitive harness evolution from slow parametric weight consolidation across four nested timescales, governed by a **4-Layer Defense-in-Depth Execution Firewall** (Gates 0–3) and a **3-Tier Shadow Evaluation Membrane**.

---

## Key Empirical Results

### 1. 30-Generation Open-Ended MAP-Elites Discovery
Across 189 commodity API evaluations with zero base-weight gradient updates, the engine achieved **100% Holdout Vault Pass Rate (V = 1.0)** across all five algorithmic domain archetypes:

| Domain Archetype | Winning Elite Harness ID | Init. Pass | Final Vault Pass (V) | Norm. Tokens | Step | Pareto Score (F) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Numerical Stability** | `h_numerical_stability_init_g4_g29` | 0.4667 | **1.0000** | 1.098 | 57 | **0.894807** |
| **Combinatorial Optimization** | `h_combinatorial_optimization_init_g3` | 1.0000 | **1.0000** | 1.194 | 54 | **0.888821** |
| **Graph Routing** | `h_graph_routing_init_g2` | 1.0000 | **1.0000** | 1.242 | 52 | **0.885941** |
| **String Automata** | `h_string_automata_init_g10` | 0.0000 | **1.0000** | 1.282 | 19 | **0.883594** |
| **Dynamic Programming** | `h_dynamic_programming_init` | 1.0000 | **1.0000** | 1.666 | 30 | **0.863219** |

### 2. Track 1: Loop 4 Parametric GRPO LoRA Distillation (`Qwen2.5-Coder-1.5B-Instruct`)
Distilling N = 20 Vault-verified System-2 trajectories (`verified_trajectories_30gen.jsonl`) into `Qwen/Qwen2.5-Coder-1.5B-Instruct` via GRPO LoRA (r = 16, alpha = 32) on an NVIDIA Tesla T4 GPU under Gate-0 AST and Gate-2 `copy.deepcopy()` verification (`track1_distillation_metrics.json`):

| Evaluation Regime | Gate-0 AST Security | Holdout Vault Pass Rate (V) | Mean Token Compression (rho) | Peak Compression (rho) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot Unhinted Synthesis** | 100.0% (20/20) | 85.00% (17/20) | 2.6233x | 4.49x |
| **Contract-Specified Synthesis** | **100.0% (20/20)** | **100.00% (20/20)** | **2.8034x** | **4.91x** |

### 3. Track 2: Self-Referential Meta-Harness DAG Mutation
Applying `CausalProgramSlicer` and structural AST surgery operators to `orchestrate_candidate` inside the 3-Tier Shadow Membrane (`track2_mutation_metrics.json`):

| Orchestration Topology / Operator | Tier 1 (AST) | Tier 2 (Reg.) | Tier 3 (Pareto) | Mean Latency (ms/task) | System-2 Calls Saved | AST Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_UNMUTATED` | Pass | Pass | Baseline | 2.3670 ms | 0.0% | 87 |
| `OP_LAZY_S2_SHORT_CIRCUIT` | Pass | Pass | **Promoted** | 0.8008 ms | 85.0% | 76 |
| `OP_SPEC_HASH_MEMO_CACHE` | Pass | Pass | **Promoted** | **0.3848 ms (6.15x)** | **92.5%** | 144 |
| `OP_UNSAFE_GATE_BYPASS_DEFECT` | **Blocked** | --- | --- | 999.0000 ms | 0.0% | 39 |

---

## Citation

```bibtex
@techreport{gavares2026bilevel,
  title={Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines},
  author={Gavares, Yasemin and Gavares, Alexander},
  institution={Autonomous Agent Architecture Research Group, Local Service App Inc / Kronoz Technologies LLC},
  year={2026}
}
