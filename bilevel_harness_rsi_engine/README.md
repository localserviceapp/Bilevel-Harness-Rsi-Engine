# Bi-Level Harness Optimization & Textual Backpropagation (4-Loop RSI Engine v4 — Open-Ended Discovery Edition)

Reference implementation for **Bi-Level Harness Optimization and Textual Backpropagation in Autonomous 4-Loop Recursive Self-Improvement Engines** (Gavares & Gavares, 2026).

## What's New in v4 (Open-Ended Discovery Upgrade)
1. **Autonomous Generative Challenger with Asymmetric ($P \neq NP$) Oracle Self-Verification (`curriculum/challenger.py`)**:
   - Synthesizes problems across **5 discovery domains** (`dynamic_programming`, `graph_routing`, `combinatorial_optimization`, `numerical_stability`, `string_automata`) or via live **Gemini 2.5 Pro** synthesis (`synthesize_autonomous_benchmark()`).
   - Prevents the **Hallucinated Oracle Trap** by passing every synthesized brute-force Reference Oracle $O_{\text{brute}}(x)$ through the **4-Layer Defense-in-Depth Firewall ($\mathcal{G}_0\text{--}\mathcal{G}_3$)** and verifying determinism, parameter liveness, and base-case consistency prior to benchmark admission.
2. **True Topological DAG & Surgical AST Mutation (`meta/meta_architect.py` & `harness/inner_harness.py`)**:
   - Loop 2 dynamically mutates the control-flow graph $G_{\text{DAG}} = (V, E)$ by injecting specialized verification and complexity-auditing nodes (`v_EdgeCaseFuzzer`, `v_ComplexityAuditor`, `v_SymmetryChecker`) into `dag_edges` and applying Python `ast.NodeTransformer` surgical AST patches (`SurgicalASTPatcher`).
3. **Continuous Multi-Epoch MAP-Elites Evolution Loop (`main.py`)**:
   - Run open-ended multi-generation discovery across domains and compute tiers with `--generations N`, tracking monotonic Pareto frontier climb ($\mathcal{A}_{\text{grid}}$), Verifier Blindspot decay ($\beta_{\text{blind}}$), and Loop 4 RLVR/GRPO Fano compression bounds ($\rho_\tau$).

## Quickstart with Gemini (Free Tier / AI Studio)
```bash
export GEMINI_API_KEY="your-api-key-here"

# Run a 5-generation open-ended discovery loop across all 5 algorithmic domains:
python3 main.py --generations 5 --provider gemini \
  --loop1-model gemini-2.5-flash \
  --loop2-model gemini-2.5-pro \
  --export-jsonl verified_trajectories.jsonl
```

## Offline Deterministic Verification Mode
```bash
python3 main.py --generations 5 --provider mock
python3 -m unittest discover -s tests -v
```
