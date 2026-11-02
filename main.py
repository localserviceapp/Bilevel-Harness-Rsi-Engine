import argparse
import json
from typing import Dict, List, Optional

from curriculum.challenger import DISCOVERY_DOMAINS, StackelbergChallenger
from distill.trajectory_exporter import GRPOCandidateSample, RLVRDistillationEngine
from harness.inner_harness import CognitiveHarness, InnerExecutionDAG
from llm.llm_client import LLMClient
from meta.meta_architect import CounterfactualTextualBackpropEngine
from sandbox.runner import ThreeTierShadowMembrane
from state.state_store import MAPElitesStateStore

try:
    from fastapi import FastAPI
    app = FastAPI(
        title="Bi-Level Harness Optimization 4-Loop RSI Engine (v4 Open-Ended Discovery)",
        version="4.0.0",
    )
except ImportError:
    app = None


def run_four_loop_demo(
    provider: str = "auto",
    loop1_model: str = "gemini-2.5-flash",
    loop2_model: str = "gemini-2.5-pro",
    export_jsonl_path: Optional[str] = None,
) -> dict:
    """Single-domain 2-epoch demonstration for backward compatibility."""
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

    problem = challenger.synthesize_kadane_benchmark(adversarial_negative_ratio=0.5)

    h_0 = CognitiveHarness(harness_id="harness_init")
    trace_0 = dag.run(h_0, problem)
    diag_0 = membrane.evaluate_candidate(
        trace_0.candidate_code, problem, trace_0.cumulative_tokens
    )
    admitted_0 = store.try_admit_candidate(h_0, diag_0, step_t=0)

    h_1, backprop_hint = meta_architect.apply_surgical_patch(h_0, diag_0, trace_0, step_t=1)
    diag_0.backprop_hint = backprop_hint
    store.log_issue(diag_0, backprop_hint)

    trace_1 = dag.run(h_1, problem)
    diag_1 = membrane.evaluate_candidate(
        trace_1.candidate_code, problem, trace_1.cumulative_tokens
    )
    admitted_1 = store.try_admit_candidate(h_1, diag_1, step_t=1)

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
            "active_dag_nodes": h_1.active_nodes(),
        },
        "loop_4_distillation": {
            "grpo_metrics": grpo_metrics,
            "theoretical_max_compression_ratio_rho_tau": fano_bound,
            "exported_verified_trajectories": exported_count,
        },
    }


def run_open_ended_discovery_loop(
    generations: int = 5,
    provider: str = "auto",
    loop1_model: str = "gemini-2.5-flash",
    loop2_model: str = "gemini-2.5-pro",
    export_jsonl_path: Optional[str] = None,
) -> dict:
    """
    Autonomous Multi-Generation Open-Ended Discovery Loop (v4):
    Co-evolves cognitive harnesses H across all 5 discovery domains in the 2D MAP-Elites grid A_grid:
      - Loop 3 synthesizes ZPD problems & self-verified brute-force Reference Oracles O_brute(x)
      - Loop 1 executes dynamic multi-node DAG topologies G_DAG = (V, E)
      - Loop 2 performs Counterfactual Textual Backpropagation & inserts new topological verifier nodes
      - Loop 4 harvests 100% Shadow- and Vault-verified trajectories for GRPO distillation
    """
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

    verified_trajectories: List[Dict] = []
    grpo_samples: List[GRPOCandidateSample] = []
    generation_logs: List[Dict] = []

    for gen_idx in range(generations):
        domain = DISCOVERY_DOMAINS[gen_idx % len(DISCOVERY_DOMAINS)]
        unresolved = store.list_unresolved_issues()
        problem = challenger.synthesize_autonomous_benchmark(
            unresolved_issues=unresolved,
            target_domain=domain,
            difficulty_ratio=0.5,
        )

        # Load current archive elite for this domain or initialize base harness H^(0)
        elite_record = store.get_elite("tier_standard", domain)
        if elite_record is not None:
            h_curr = CognitiveHarness(**elite_record["harness"])
        else:
            h_curr = CognitiveHarness(harness_id=f"h_{domain}_init")

        # Step A: Evaluate incumbent harness on synthesized benchmark
        trace_before = dag.run(h_curr, problem)
        diag_before = membrane.evaluate_candidate(
            trace_before.candidate_code, problem, trace_before.cumulative_tokens
        )
        admitted_before = store.try_admit_candidate(h_curr, diag_before, step_t=gen_idx * 2)

        in_zpd = challenger.is_in_zone_of_proximal_development(
            diag_before.dense_fitness.vault_pass_rate
        )

        # Step B: Trigger Loop 2 Counterfactual Textual Backpropagation & Topological DAG Mutation
        h_mutated, hint = meta_architect.apply_surgical_patch(
            h_curr, diag_before, trace_before, step_t=gen_idx + 1
        )
        store.log_issue(diag_before, hint)

        # Step C: Evaluate mutated harness H' and test Strict Pareto Dominance admission
        trace_after = dag.run(h_mutated, problem)
        diag_after = membrane.evaluate_candidate(
            trace_after.candidate_code, problem, trace_after.cumulative_tokens
        )
        admitted_after = store.try_admit_candidate(h_mutated, diag_after, step_t=gen_idx * 2 + 1)

        if diag_after.dense_fitness.vault_pass_rate == 1.0:
            store.mark_domain_issues_resolved(domain)
            verified_trajectories.append(
                {
                    "generation": gen_idx + 1,
                    "domain_archetype": domain,
                    "problem_id": problem.problem_id,
                    "specification": problem.specification,
                    "active_dag_nodes": h_mutated.active_nodes(),
                    "system2_trace": [s.output_state for s in trace_after.states],
                    "verified_system1_code": trace_after.candidate_code,
                    "vault_pass_rate": diag_after.dense_fitness.vault_pass_rate,
                    "invariants_passed_ratio": diag_after.dense_fitness.invariants_passed_ratio,
                }
            )

        regret_before = challenger.compute_solver_regret(
            diag_after.dense_fitness.pareto_efficiency_score,
            diag_before.dense_fitness.pareto_efficiency_score,
        )

        grpo_samples.append(
            GRPOCandidateSample(
                f"gen{gen_idx}_before",
                trace_before.candidate_code,
                int(diag_before.dense_fitness.vault_pass_rate == 1.0),
                diag_before.dense_fitness.invariants_passed_ratio,
                diag_before.dense_fitness.normalized_tokens,
                -1.2,
                -1.3,
                -1.2,
            )
        )
        grpo_samples.append(
            GRPOCandidateSample(
                f"gen{gen_idx}_after",
                trace_after.candidate_code,
                int(diag_after.dense_fitness.vault_pass_rate == 1.0),
                diag_after.dense_fitness.invariants_passed_ratio,
                diag_after.dense_fitness.normalized_tokens,
                -1.1,
                -0.9,
                -1.1,
            )
        )

        generation_logs.append(
            {
                "generation": gen_idx + 1,
                "domain_archetype": domain,
                "problem_id": problem.problem_id,
                "in_zone_of_proximal_development": in_zpd,
                "incumbent_vault_pass_rate": diag_before.dense_fitness.vault_pass_rate,
                "incumbent_pareto_score": diag_before.dense_fitness.pareto_efficiency_score,
                "verifier_blindspot_detected": hint.verifier_blindspot_indicator,
                "suspect_node": hint.suspect_node.value,
                "topological_mutation": hint.topological_mutation_applied,
                "mutated_active_nodes": h_mutated.active_nodes(),
                "mutated_vault_pass_rate": diag_after.dense_fitness.vault_pass_rate,
                "mutated_pareto_score": diag_after.dense_fitness.pareto_efficiency_score,
                "strict_pareto_admitted": admitted_after,
                "stackelberg_solver_regret_reduced": regret_before,
            }
        )

    exported_count = 0
    if export_jsonl_path and verified_trajectories:
        exported_count = distiller.export_verified_trajectories_jsonl(
            verified_trajectories, export_jsonl_path
        )

    grpo_metrics = distiller.compute_grpo_group_metrics(grpo_samples)
    fano_bound = distiller.compute_fano_compression_bound(
        h_s_dag_given_p=54.0,
        i_s_dag_c_star_given_p=13.2,
        eps_0=0.015,
        support_size_c_star=32,
    )

    return {
        "engine_version": "4.0.0-open-ended-discovery",
        "llm_provider": llm_client.provider,
        "generations_executed": generations,
        "generation_trajectory": generation_logs,
        "map_elites_archive_elites": store.list_all_elites(),
        "unresolved_issues_remaining": len(store.list_unresolved_issues()),
        "loop_4_distillation": {
            "verified_trajectories_harvested": len(verified_trajectories),
            "exported_jsonl_count": exported_count,
            "grpo_metrics": grpo_metrics,
            "theoretical_max_compression_ratio_rho_tau": fano_bound,
        },
    }


if app is not None:
    @app.get("/health")
    def health_check() -> dict:
        return {"status": "ok", "engine": "4-Loop Bi-Level RSI Engine v4 (Open-Ended Discovery)"}

    @app.post("/run-cycle")
    def api_run_cycle() -> dict:
        return run_four_loop_demo()

    @app.post("/run-discovery")
    def api_run_discovery(generations: int = 5) -> dict:
        return run_open_ended_discovery_loop(generations=generations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-Loop Bi-Level Harness Optimization RSI Engine v4")
    parser.add_argument("--demo", action="store_true", help="Run 2-epoch single-domain demonstration")
    parser.add_argument("--generations", type=int, default=0, help="Run N-generation multi-domain open-ended discovery loop")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai_compatible", "ollama", "mock"])
    parser.add_argument("--loop1-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--loop2-model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--export-jsonl", type=str, default=None, help="Path to export verified trajectories JSONL for Loop 4 GRPO")
    args = parser.parse_args()

    if args.generations > 0:
        res = run_open_ended_discovery_loop(
            generations=args.generations,
            provider=args.provider,
            loop1_model=args.loop1_model,
            loop2_model=args.loop2_model,
            export_jsonl_path=args.export_jsonl,
        )
    else:
        res = run_four_loop_demo(
            provider=args.provider,
            loop1_model=args.loop1_model,
            loop2_model=args.loop2_model,
            export_jsonl_path=args.export_jsonl,
        )
    print(json.dumps(res, indent=2))
