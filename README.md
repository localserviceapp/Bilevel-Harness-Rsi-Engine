# Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23935034.svg)](https://doi.org/10.5281/zenodo.23935034)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Authors:** Yasemin Gavares and Alexander Gavares  
**Affiliations:** Autonomous Agent Architecture Research Group / Local Service App Inc / Kronoz Technologies LLC  

Official reference implementation for the **Autonomous 4-Loop Recursive Self-Improvement (RSI)** engine governed by an airtight **4-Layer Defense-in-Depth Execution Substrate** ($\mathcal{G}_0$–$\mathcal{G}_3$). Instead of modifying foundational model weights during inner-loop search, an outer meta-optimization loop searches over a structured cognitive harness space $\mathcal{H} \in \mathcal{S}_{\mathcal{H}}$ via Causal Program Slicing and Counterfactual Textual Backpropagation over Directed Acyclic Execution Graphs (DAGs).

---

## 1. System Topology: The 4-Loop Architecture

The framework unrolls recursive self-improvement across four distinct temporal timescales ($\Delta t_1 \ll \Delta t_2 \ll \Delta t_3 \ll \Delta t_4$) to decouple inner-loop inference from outer-loop structural and parametric evolution:

* **Loop 1 — Inner Execution DAG ($\Delta t_1 \sim 10^1\text{ s}$):** Multi-node execution pipeline (`harness/inner_harness.py`) that synthesizes and refines executable Python candidates against Tier 1 Public Tests and runtime contracts.
* **Loop 2 — Meta-Architect Outer Loop ($\Delta t_2 \sim 10^3\text{ s}$):** Performs Causal Program Slicing and Counterfactual Textual Backpropagation over execution traces to mutate control-flow topology, refine node system prompts, and extract reusable verified primitives into `stdlib_primitives.py`.
* **Loop 3 — Adversarial Challenger ($\Delta t_3 \sim 10^5\text{ s}$):** Synthesizes verifiable benchmark problems at the Frontier of Competence and generates property-based Hypothesis fuzzers via Stackelberg regret-maximization.
* **Loop 4 — Parametric Distillation ($\Delta t_4 \sim 10^6\text{ s}$):** Compresses verified multi-step System-2 execution trajectories into single-step System-1 intuition via Group Relative Policy Optimization (GRPO) over domain-indexed LoRA adapters ($\Delta \theta_d$) while keeping the foundational trunk frozen.

---

## 2. 4-Layer Defense-in-Depth Execution Substrate ($\mathcal{G}_0$–$\mathcal{G}_3$)

* **$\mathcal{G}_0$ (Static AST & Complexity Gate):** Enforces a strict zero-reflection ban ($\Omega_{\text{forbidden}}$ blocking `eval`, `exec`, dynamic imports, and frame introspection attributes such as `tb_frame`, `gi_frame`, and `globals`) alongside Halstead volume bounds.
* **$\mathcal{G}_1$ (CPython Kernel Security Hook):** Attaches an unremovable PEP 578 runtime audit hook (`sys.addaudithook`) at the C-API layer to intercept unauthorized subprocess, network socket, or filesystem operations.
* **$\mathcal{G}_2$ (State & Memory Isolation):** Enforces recursive deep-copy input isolation (`copy.deepcopy()`) prior to candidate execution to prevent in-place reference mutation across test tiers.
* **$\mathcal{G}_3$ (Ephemeral Asymmetric Sandbox):** Executes untrusted candidates inside a network-disabled, read-only container governed by strict `cgroups v2` limits ($\text{CPU}=1.0$, $\text{RAM}=512\text{ MB}$).

---

## 3. Empirical Findings (30-Generation Open-Ended Discovery Run)

We evaluated the engine across a 30-generation open-ended discovery run on commodity LLM endpoints using our sticky multi-model Gemini Flash failover router (`llm/llm_client.py`):

* **100% Holdout Vault Convergence:** Achieved a $100\%$ holdout Vault pass rate ($V = 1.0000$) across all five benchmark algorithmic domain archetypes in the Two-Timescale Quality-Diversity MAP-Elites archive.
* **Numerical Stability Breakthrough (`kahan_sum_zpd_50`):** Overcame severe floating-point cancellation where baseline harnesses stalled at Generation 9 ($V = 0.4667$). Autonomous structural DAG surgery spliced in `v_SymmetryChecker`, driving convergence to the archive's highest Pareto fitness score of **$0.8948$** in Generation 29.
* **Autonomous Structural DAG Surgery:** Detected $6$ verifier blindspots ($\beta_{\text{blind}} = 1$, where Tier 1 public tests passed but Tier 2/3 holdout vaults failed) and autonomously dispatched topological graph-edit operators (`v_SymmetryChecker` and `v_EdgeCaseFuzzer`) without human intervention.
* **Information-Theoretic Policy Compression:** Loop 4 RLVR/GRPO trajectory distillation yielded an empirical policy token compression ratio of **$\rho_{\tau} = 4.1498$**.

| Domain Archetype | Representative Benchmark | Initial Vault Pass Rate | Final Vault Pass Rate ($V$) | Peak Pareto Score | Autonomous DAG Surgery Triggered |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Numerical Stability** | `kahan_sum_zpd_50` | $0.4667$ | **$1.0000$** | **$0.8948$** | `v_SymmetryChecker` (Gen 29) |
| **Combinatorial Optimization** | `knapsack_branch_bound` | $0.6000$ | **$1.0000$** | **$0.8712$** | `v_EdgeCaseFuzzer` |
| **Graph Routing** | `constrained_shortest_path` | $0.5333$ | **$1.0000$** | **$0.8645$** | `v_SymmetryChecker` |
| **String Automata** | `regex_nfa_compiler` | $0.6667$ | **$1.0000$** | **$0.8810$** | `v_EdgeCaseFuzzer` |
| **Dynamic Programming** | `interval_schedule_pareto` | $0.7333$ | **$1.0000$** | **$0.8894$** | Verified Primitive Extraction |

---

## 4. Resilient Multi-Model Gemini Flash Failover Router

To prevent API rate limits (`HTTP 429`), transient server overloads (`HTTP 503`), or deprecated model endpoints (`HTTP 404`) from interrupting multi-generation evolutionary runs, `llm/llm_client.py` implements:
1. **Sticky Endpoint Pointer:** Locks onto the fastest responding Gemini Flash model until a rate limit is encountered.
2. **Automatic 404 Blacklisting:** Immediately prunes unavailable model identifiers from the active rotation pool.
3. **Exponential Backoff Sweep:** Rotates seamlessly across fallback Flash endpoints with bounded exponential backoff and a 180-second reasoning socket timeout.

---

## 5. Quickstart

```bash
export GEMINI_API_KEY="your_api_key_here"
python3 main.py

@techreport{gavares2026bilevel,
  title       = {Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines},
  author      = {Gavares, Yasemin and Gavares, Alexander},
  year        = {2026},
  institution = {Autonomous Agent Architecture Research Group / Local Service App Inc / Kronoz Technologies LLC},
  type        = {Technical Note},
  doi         = {10.5281/zenodo.23935034},
  url         = {[https://doi.org/10.5281/zenodo.23935034](https://doi.org/10.5281/zenodo.23935034)}
}
