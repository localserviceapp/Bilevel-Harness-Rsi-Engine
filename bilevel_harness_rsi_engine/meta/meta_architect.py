import ast
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


class SurgicalASTPatcher(ast.NodeTransformer):
    """
    Applies surgical Abstract Syntax Tree (AST) transformations to candidate Python code
    and verifies syntactic soundness via ast.parse / ast.unparse.
    """

    def __init__(self, patch_type: ASTPatchType):
        super().__init__()
        self.patch_type = patch_type

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        if self.patch_type in {
            ASTPatchType.INJECT_BOUNDARY_GUARD,
            ASTPatchType.STRENGTHEN_PRECONDITION_ASSERTION,
            ASTPatchType.INSERT_EDGE_CASE_FUZZER_NODE,
        } and node.args.args:
            first_arg = node.args.args[0].arg
            # Check if an empty-input guard is already present at top of function body
            has_guard = False
            if node.body and isinstance(node.body[0], ast.If):
                has_guard = True
            if not has_guard:
                guard_stmt = ast.parse(
                    f"if {first_arg} is None:\n    return 0\n"
                ).body[0]
                node.body.insert(0, guard_stmt)
        return node

    @classmethod
    def transform_code(cls, code: str, patch_type: ASTPatchType) -> str:
        try:
            tree = ast.parse(code)
            patched_tree = cls(patch_type).visit(tree)
            ast.fix_missing_locations(patched_tree)
            if hasattr(ast, "unparse"):
                return ast.unparse(patched_tree) + "\n"
            return code
        except Exception:
            return code


class CounterfactualTextualBackpropEngine:
    """
    Loop 2 (Meta-Architect Outer Loop, dt_2 ~ 10^3 s):
    Formalizes the Textual Backpropagation Operator nabla_H^text using Verifier Blindspot Gating,
    Counterfactual State Sensitivity, Surgical AST Patching, and Dynamic Topological DAG Mutation.
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
        active_node_names = [s.node_name for s in trace.states]
        if not active_node_names:
            active_node_names = [
                SuspectNode.PLANNER.value,
                SuspectNode.CODER.value,
                SuspectNode.LOCAL_VERIFIER.value,
            ]

        if self.llm_client is not None and self.llm_client.is_live:
            try:
                critic_prompt = (
                    "Evaluate semantic defect alignment psi_diag(s_v, e_struct) >= 0 for each DAG node.\n"
                    f"Active Nodes: {active_node_names}\n"
                    f"Failure Taxonomy: {diagnostic.structural_failure.failure_taxonomy.value}\n"
                    f"Sanitized Reason: {diagnostic.structural_failure.sanitized_reason}\n"
                    f"Public Pass Rate: {diagnostic.public_pass_rate}, "
                    f"Shadow Invariants Ratio: {diagnostic.dense_fitness.invariants_passed_ratio}\n"
                    "Return JSON mapping each node name to a non-negative float."
                )
                resp = self.llm_client.generate(
                    prompt=critic_prompt,
                    system_instruction="You are the Loop 2 Meta-Architect Critic. Output only valid JSON.",
                    model=self.llm_client.loop2_model,
                    json_mode=True,
                )
                parsed = extract_json_object(resp.text)
                if any(k in parsed for k in active_node_names):
                    return {
                        k: max(0.0, float(parsed.get(k, 0.5)))
                        for k in active_node_names
                    }
            except Exception:
                pass

        taxonomy = diagnostic.structural_failure.failure_taxonomy
        scores = {k: 0.5 for k in active_node_names}

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
        patch_type = self._select_patch_type(
            suspect,
            diagnostic.structural_failure.failure_taxonomy,
            beta_blind,
            diagnostic.domain_archetype,
        )

        return BackpropHint(
            suspect_node=suspect,
            ast_patch_type=patch_type,
            verifier_blindspot_indicator=beta_blind,
            node_credit_weights=alphas,
        )

    def _select_patch_type(
        self,
        suspect: SuspectNode,
        taxonomy: FailureTaxonomy,
        beta_blind: int,
        domain_archetype: str,
    ) -> ASTPatchType:
        if taxonomy == FailureTaxonomy.NONE:
            return ASTPatchType.NONE
        if beta_blind == 1 or suspect == SuspectNode.LOCAL_VERIFIER:
            if domain_archetype in {"combinatorial_optimization", "graph_routing"}:
                return ASTPatchType.INSERT_COMPLEXITY_AUDITOR_NODE
            if domain_archetype in {"string_automata", "numerical_stability"}:
                return ASTPatchType.INSERT_SYMMETRY_CHECKER_NODE
            return ASTPatchType.INSERT_EDGE_CASE_FUZZER_NODE
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
        """
        Executes both:
          1. Topological DAG Mutation on G_DAG = (V, E) (inserting v_EdgeCaseFuzzer, v_ComplexityAuditor, or v_SymmetryChecker)
          2. Prompt & Verification Policy updates via L_meta
        """
        hint = self.compute_credit_assignment(trace, diagnostic)
        mutated = copy.deepcopy(harness)
        mutated.harness_id = f"{harness.harness_id}_g{step_t}"

        target_key = hint.suspect_node.value
        topo_mutation_desc = None

        if hint.verifier_blindspot_indicator == 1:
            mutated.verification_policies["enforce_boundary_assertions"] = True
            mutated.verification_policies["enforce_empty_and_negative_checks"] = True

            # Insert new topological node into G_DAG = (V, E) based on patch type
            if hint.ast_patch_type == ASTPatchType.INSERT_EDGE_CASE_FUZZER_NODE:
                new_node = SuspectNode.EDGE_CASE_FUZZER.value
                mutated.verification_policies["enable_edge_case_fuzzer_node"] = True
                mutated.prompts[new_node] = (
                    "Synthesize extremal negative, zero, and empty boundary inputs to falsify naive candidates."
                )
            elif hint.ast_patch_type == ASTPatchType.INSERT_COMPLEXITY_AUDITOR_NODE:
                new_node = SuspectNode.COMPLEXITY_AUDITOR.value
                mutated.verification_policies["enable_complexity_auditor_node"] = True
                mutated.prompts[new_node] = (
                    "Verify dynamic programming optimal substructure and reject greedy counterexamples."
                )
            else:
                new_node = SuspectNode.SYMMETRY_CHECKER.value
                mutated.verification_policies["enable_symmetry_checker_node"] = True
                mutated.prompts[new_node] = (
                    "Verify numerical Kahan compensation and even/odd palindrome symmetry invariants."
                )

            if ["v_Coder", new_node] not in mutated.dag_edges:
                mutated.dag_edges.append(["v_Coder", new_node])
            if [new_node, "v_LocalVerifier"] not in mutated.dag_edges:
                mutated.dag_edges.append([new_node, "v_LocalVerifier"])
            topo_mutation_desc = f"Inserted node {new_node} into G_DAG edges: v_Coder -> {new_node} -> v_LocalVerifier"

            mutated.prompts[target_key] = (
                mutated.prompts.get(target_key, "")
                + f" [PATCH: Route verification through {new_node} prior to admission.]"
            )
            mutated.prompts["v_Coder"] = (
                mutated.prompts.get("v_Coder", "")
                + " [PATCH: Enforce exact DP recurrence / Kahan compensation / full boundary handling.]"
            )
        elif hint.ast_patch_type == ASTPatchType.INJECT_BOUNDARY_GUARD:
            mutated.verification_policies["enforce_boundary_assertions"] = True
            mutated.prompts[target_key] = mutated.prompts.get(target_key, "") + " [PATCH: Inject explicit boundary guard.]"
            topo_mutation_desc = "Applied AST boundary guard transformation."
        elif hint.ast_patch_type == ASTPatchType.ELIMINATE_FORBIDDEN_REFLECTION:
            mutated.prompts[target_key] = mutated.prompts.get(target_key, "") + " [PATCH: Strip reflection and in-place mutation.]"
            topo_mutation_desc = "Applied AST zero-reflection filter."

        hint.topological_mutation_applied = topo_mutation_desc
        return mutated, hint
