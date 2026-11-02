import argparse
import json
from typing import Optional

from curriculum.challenger import StackelbergChallenger
from distill.trajectory_exporter import GRPOCandidateSample, RLVRDistillationEngine
from harness.inner_harness import CognitiveHarness, InnerExecutionDAG
from llm.llm_client import LLMClient
from meta.meta_architect import CounterfactualTextualBackpropEngine
from sandbox.runner import ThreeTierShadowMembrane
from state.state_store import MAPElitesStateStore

try:
    from fastapi import FastAPI
    app = FastAPI(
        title="Bi-Level Harness Optimization 4-Loop RSI Engine",
        version="3.0.0",
    )
except ImportError:
    app = None


def run_four_loop_demo(
    provider: str = "auto",
    loop1_model: str = "gemini-2.5-flash",
    loop2_model: str = "gemini-2.5-pro",
    export_jsonl_path: Optional[str] = None,
) -> dict:
    llm_client = LLMClient(
        provider=provider,
        loop1_model=loop1_model,
        loop2_model=loop2_model,
    )
    store = MAPElitesStateStore(":memory:")
    membrane = ThreeTierShadowMembrane()
    dag = InnerExecutionDAG(llm_client=llm_client)
    meta_architect = CounterfactualTextualBackpropEngine(lambda_verifier=3.0, llm_client=llm_client)
    challenger = StackelbergChallenger(llm_client=llm_client)
    distiller = RLVRDistillationEngine()

    # Loop 3: Synthesize benchmark in Zone of Proximal Development
    problem = challenger.synthesize_kadane_benchmark(adversarial_negative_ratio=0.5)

    # Loop 1 (Epoch 0): Run initial harness H^(0)
    h_0 = CognitiveHarness(harness_id="harness_init")
    trace_0 = dag.run(h_0, problem)
    diag_0 = membrane.evaluate_candidate(
        trace_0.candidate_code, problem, trace_0.cumulative_tokens
    )
    admitted_0 = store.try_admit_candidate(h_0, diag_0, step_t=0)

    # Loop 2: Counterfactual Textual Backpropagation & Surgical Patching
    h_1, backprop_hint = meta_architect.apply_surgical_patch(h_0, diag_0, trace_0, step_t=1)
    diag_0.backprop_hint = backprop_hint
    store.log_issue(diag_0, backprop_hint)

    # Loop 1 (Epoch 1): Re-evaluate patched harness H^(1)
    trace_1 = dag.run(h_1, problem)
    diag_1 = membrane.evaluate_candidate(
        trace_1.candidate_code, problem, trace_1.cumulative_tokens
    )
    admitted_1 = store.try_admit_candidate(h_1, diag_1, step_t=1)

    # Loop 4: Weight Distillation & Information-Theoretic Compression Bound
    samples = [
        GRPOCandidateSample("y_1", trace_0.candidate_code, int(diag_0.dense_fitness.vault_pass_rate == 1.0), diag_0.dense_fitness.invariants_passed_ratio, diag_0.dense_fitness.normalized_tokens, -1.2, -1.3, -1.2),
        GRPOCandidateSample("y_2", trace_1.candidate_code, int(diag_1.dense_fitness.vault_pass_rate == 1.0), diag_1.dense_fitness.invariants_passed_ratio, diag_1.dense_fitness.normalized_tokens, -1.1, -0.9, -1.1),
    ]
    grpo_metrics = distiller.compute_grpo_group_metrics(samples)
    fano_bound = distiller.compute_fano_compression_bound(
        h_s_dag_given_p=48.0,
        i_s_dag_c_star_given_p=12.5,
        eps_0=0.02,
        support_size_c_star=16,
    )

    exported_count = 0
    if export_jsonl_path:
        trajectories = [
            {
                "problem_id": problem.problem_id,
                "specification": problem.specification,
                "system2_trace": [s.output_state for s in trace_1.states],
                "verified_system1_code": trace_1.candidate_code,
                "vault_pass_rate": diag_1.dense_fitness.vault_pass_rate,
                "invariants_passed_ratio": diag_1.dense_fitness.invariants_passed_ratio,
            }
        ]
        exported_count = distiller.export_verified_trajectories_jsonl(trajectories, export_jsonl_path)

    return {
        "llm_provider": llm_client.provider,
        "loop1_model": llm_client.loop1_model,
        "loop2_model": llm_client.loop2_model,
        "problem_id": problem.problem_id,
        "epoch_0": {
            "harness_id": h_0.harness_id,
            "public_pass_rate": diag_0.public_pass_rate,
            "shadow_pass_rate": diag_0.dense_fitness.invariants_passed_ratio,
            "vault_pass_rate": diag_0.dense_fitness.vault_pass_rate,
            "pareto_score": diag_0.dense_fitness.pareto_efficiency_score,
            "admitted": admitted_0,
            "backprop_hint": backprop_hint.model_dump(),
        },
        "epoch_1": {
            "harness_id": h_1.harness_id,
            "public_pass_rate": diag_1.public_pass_rate,
            "shadow_pass_rate": diag_1.dense_fitness.invariants_passed_ratio,
            "vault_pass_rate": diag_1.dense_fitness.vault_pass_rate,
            "pareto_score": diag_1.dense_fitness.pareto_efficiency_score,
            "admitted": admitted_1,
        },
        "loop_4_distillation": {
            "grpo_metrics": grpo_metrics,
            "theoretical_max_compression_ratio_rho_tau": fano_bound,
            "exported_verified_trajectories": exported_count,
        },
    }


if app is not None:
    @app.get("/health")
    def health_check() -> dict:
        return {"status": "ok", "engine": "4-Loop Bi-Level RSI Engine v3"}

    @app.post("/run-cycle")
    def api_run_cycle() -> dict:
        return run_four_loop_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-Loop Bi-Level Harness Optimization RSI Engine")
    parser.add_argument("--demo", action="store_true", help="Run 4-loop RSI demonstration cycle")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai_compatible", "ollama", "mock"])
    parser.add_argument("--loop1-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--loop2-model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--export-jsonl", type=str, default=None, help="Path to export verified trajectories JSONL for Loop 4 GRPO")
    args = parser.parse_args()
    res = run_four_loop_demo(
        provider=args.provider,
        loop1_model=args.loop1_model,
        loop2_model=args.loop2_model,
        export_jsonl_path=args.export_jsonl,
    )
    print(json.dumps(res, indent=2))
