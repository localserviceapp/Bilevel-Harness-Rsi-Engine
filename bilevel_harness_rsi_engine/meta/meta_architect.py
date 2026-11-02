import copy
import math
from typing import Dict, Optional, Tuple

from harness.inner_harness import CognitiveHarness, ExecutionTrace
from llm.llm_client import LLMClient, extract_json_object
from schemas.diagnostic_contract import (
    ASTPatchType,
    BackpropHint,
    FailureTaxonomy,
    SanitizedDiagnosticPayload,
    SuspectNode,
)


class CounterfactualTextualBackpropEngine:
    """
    Loop 2 (Meta-Architect Outer Loop, dt_2 ~ 10^3 s):
    Formalizes the Textual Backpropagation Operator nabla_H^text using Verifier Blindspot Gating
    and Counterfactual State Sensitivity. Supports live LLM critic scoring (e.g. Gemini 2.5 Pro).
    """

    def __init__(self, lambda_verifier: float = 3.0, llm_client: Optional[LLMClient] = None):
        self.lambda_verifier = lambda_verifier
        self.llm_client = llm_client

    def compute_verifier_blindspot(
        self, local_verifier_passed: bool, shadow_invariants_ratio: float
    ) -> int:
        return int(bool(local_verifier_passed) and (shadow_invariants_ratio < 1.0))

    def evaluate_semantic_defect_alignment(
        self, trace: ExecutionTrace, diagnostic: SanitizedDiagnosticPayload
    ) -> Dict[str, float]:
        if self.llm_client is not None and self.llm_client.is_live:
            try:
                critic_prompt = (
                    "Evaluate semantic defect alignment psi_diag(s_v, e_struct) >= 0 for each DAG node.\n"
                    f"Failure Taxonomy: {diagnostic.structural_failure.failure_taxonomy.value}\n"
                    f"Sanitized Reason: {diagnostic.structural_failure.sanitized_reason}\n"
                    f"Public Pass Rate: {diagnostic.public_pass_rate}, "
                    f"Shadow Invariants Ratio: {diagnostic.dense_fitness.invariants_passed_ratio}\n"
                    "Return JSON with float keys: {\"v_Planner\": float, \"v_Coder\": float, \"v_LocalVerifier\": float}."
                )
                resp = self.llm_client.generate(
                    prompt=critic_prompt,
                    system_instruction="You are the Loop 2 Meta-Architect Critic. Output only valid JSON.",
                    model=self.llm_client.loop2_model,
                    json_mode=True,
                )
                parsed = extract_json_object(resp.text)
                if all(k in parsed for k in ("v_Planner", "v_Coder", "v_LocalVerifier")):
                    return {
                        "v_Planner": max(0.0, float(parsed["v_Planner"])),
                        "v_Coder": max(0.0, float(parsed["v_Coder"])),
                        "v_LocalVerifier": max(0.0, float(parsed["v_LocalVerifier"])),
                    }
            except Exception:
                pass

        taxonomy = diagnostic.structural_failure.failure_taxonomy
        scores = {
            SuspectNode.PLANNER.value: 0.5,
            SuspectNode.CODER.value: 0.5,
            SuspectNode.LOCAL_VERIFIER.value: 0.5,
        }

        if taxonomy in {FailureTaxonomy.SANDBOX_VIOLATION, FailureTaxonomy.REWARD_HACK, FailureTaxonomy.IN_PLACE_MUTATION}:
            scores[SuspectNode.CODER.value] = 3.2
            scores[SuspectNode.PLANNER.value] = 1.1
        elif taxonomy == FailureTaxonomy.SYNTAX_OR_RUNTIME_ERROR:
            scores[SuspectNode.CODER.value] = 2.8
            scores[SuspectNode.LOCAL_VERIFIER.value] = 1.4
        elif taxonomy == FailureTaxonomy.INVARIANT_VIOLATION:
            scores[SuspectNode.PLANNER.value] = 1.9
            scores[SuspectNode.CODER.value] = 2.1
            scores[SuspectNode.LOCAL_VERIFIER.value] = 1.6

        return scores

    def compute_credit_assignment(
        self, trace: ExecutionTrace, diagnostic: SanitizedDiagnosticPayload
    ) -> BackpropHint:
        beta_blind = self.compute_verifier_blindspot(
            trace.local_verifier_passed, diagnostic.dense_fitness.invariants_passed_ratio
        )
        psi_scores = self.evaluate_semantic_defect_alignment(trace, diagnostic)

        logits: Dict[str, float] = {}
        for node_name, psi_val in psi_scores.items():
            is_verifier = 1.0 if node_name == SuspectNode.LOCAL_VERIFIER.value else 0.0
            logits[node_name] = psi_val + self.lambda_verifier * beta_blind * is_verifier

        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        denom = sum(exp_logits.values())
        alphas = {k: round(val / denom, 6) for k, val in exp_logits.items()}
        top_node_str = max(alphas, key=alphas.get)
        alphas[top_node_str] = round(alphas[top_node_str] + (1.0 - sum(alphas.values())), 6)

        suspect = SuspectNode(top_node_str)
        patch_type = self._select_patch_type(suspect, diagnostic.structural_failure.failure_taxonomy, beta_blind)

        return BackpropHint(
            suspect_node=suspect,
            ast_patch_type=patch_type,
            verifier_blindspot_indicator=beta_blind,
            node_credit_weights=alphas,
        )

    def _select_patch_type(
        self, suspect: SuspectNode, taxonomy: FailureTaxonomy, beta_blind: int
    ) -> ASTPatchType:
        if taxonomy == FailureTaxonomy.NONE:
            return ASTPatchType.NONE
        if beta_blind == 1 or suspect == SuspectNode.LOCAL_VERIFIER:
            return ASTPatchType.STRENGTHEN_PRECONDITION_ASSERTION
        if taxonomy in {FailureTaxonomy.SANDBOX_VIOLATION, FailureTaxonomy.REWARD_HACK}:
            return ASTPatchType.ELIMINATE_FORBIDDEN_REFLECTION
        if suspect == SuspectNode.CODER:
            return ASTPatchType.INJECT_BOUNDARY_GUARD
        return ASTPatchType.REWRITE_ALGORITHMIC_DECOMPOSITION

    def apply_surgical_patch(
        self,
        harness: CognitiveHarness,
        diagnostic: SanitizedDiagnosticPayload,
        trace: ExecutionTrace,
        step_t: int,
    ) -> Tuple[CognitiveHarness, BackpropHint]:
        hint = self.compute_credit_assignment(trace, diagnostic)
        mutated = copy.deepcopy(harness)
        mutated.harness_id = f"{harness.harness_id}_v{step_t}"

        target_key = hint.suspect_node.value
        if hint.verifier_blindspot_indicator == 1:
            mutated.verification_policies["enforce_boundary_assertions"] = True
            mutated.verification_policies["enforce_empty_and_negative_checks"] = True
            mutated.prompts[target_key] = (
                mutated.prompts.get(target_key, "")
                + " [PATCH: Enforce property-based negative/empty boundary invariants prior to admission.]"
            )
            mutated.prompts["v_Coder"] = (
                mutated.prompts.get("v_Coder", "")
                + " [PATCH: Initialize DP state from first element instead of 0 to handle negative inputs.]"
            )
        elif hint.ast_patch_type == ASTPatchType.INJECT_BOUNDARY_GUARD:
            mutated.verification_policies["enforce_boundary_assertions"] = True
            mutated.prompts[target_key] += " [PATCH: Inject explicit boundary guard.]"
        elif hint.ast_patch_type == ASTPatchType.ELIMINATE_FORBIDDEN_REFLECTION:
            mutated.prompts[target_key] += " [PATCH: Strip reflection and in-place mutation.]"

        return mutated, hint
