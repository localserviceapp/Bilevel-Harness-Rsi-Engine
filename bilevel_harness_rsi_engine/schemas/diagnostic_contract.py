from enum import Enum
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class FailureTaxonomy(str, Enum):
    NONE = "NONE"
    REWARD_HACK = "REWARD_HACK"
    SANDBOX_VIOLATION = "SANDBOX_VIOLATION"
    IN_PLACE_MUTATION = "IN_PLACE_MUTATION"
    SYNTAX_OR_RUNTIME_ERROR = "SYNTAX_OR_RUNTIME_ERROR"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    TIMEOUT_OR_OOM = "TIMEOUT_OR_OOM"


class SuspectNode(str, Enum):
    PLANNER = "v_Planner"
    CODER = "v_Coder"
    LOCAL_VERIFIER = "v_LocalVerifier"


class ASTPatchType(str, Enum):
    NONE = "NONE"
    STRENGTHEN_PRECONDITION_ASSERTION = "STRENGTHEN_PRECONDITION_ASSERTION"
    INJECT_BOUNDARY_GUARD = "INJECT_BOUNDARY_GUARD"
    REWRITE_ALGORITHMIC_DECOMPOSITION = "REWRITE_ALGORITHMIC_DECOMPOSITION"
    ELIMINATE_FORBIDDEN_REFLECTION = "ELIMINATE_FORBIDDEN_REFLECTION"


class GateTelemetry(BaseModel):
    """Track A: Execution Firewall Telemetry (Gates G0 - G3)."""
    triggered_gate: Optional[Literal["G0", "G1", "G2", "G3"]] = Field(
        default=None, description="First defense-in-depth gate that short-circuited execution, or None if all passed."
    )
    cyclomatic_complexity: float = Field(default=1.0, ge=0.0)
    halstead_volume: float = Field(default=0.0, ge=0.0)
    audit_hook_tripped: bool = Field(default=False)
    violated_identifiers: List[str] = Field(default_factory=list)


class StructuralFailure(BaseModel):
    """Track A: Scrubbed structural failure signal e_struct."""
    failure_taxonomy: FailureTaxonomy = Field(default=FailureTaxonomy.NONE)
    oracle_divergence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of shadow/vault tests diverging from Reference Oracle O(x)."
    )
    sanitized_reason: str = Field(
        default="OK", description="Scrubbed diagnostic description with zero raw test fixture leakage."
    )


class DenseFitnessVec(BaseModel):
    """Track B: Multi-objective fitness & regularized Pareto score."""
    vault_pass_rate: float = Field(..., ge=0.0, le=1.0, description="V(H, p) in [0, 1]")
    normalized_tokens: float = Field(..., ge=0.0, description="tau_bar(H, p) = tau / tau_0")
    normalized_latency: float = Field(..., ge=0.0, description="mu_bar(H, p) = mu / mu_0")
    pareto_efficiency_score: float = Field(..., ge=0.0, le=1.0, description="F(H, p) in [0, 1]")
    invariants_passed_ratio: float = Field(..., ge=0.0, le=1.0, description="psi_sh(H, p) in [0, 1]")
    p95_latency_ms: float = Field(..., ge=0.0)

    def objective_vector(self) -> List[float]:
        """Returns J(H; P_d) = [V(H, p), -tau_bar(H, p), -mu_bar(H, p)]."""
        return [self.vault_pass_rate, -self.normalized_tokens, -self.normalized_latency]


class BackpropHint(BaseModel):
    """Track B: Counterfactual Textual Backpropagation routing hint."""
    suspect_node: SuspectNode
    ast_patch_type: ASTPatchType
    verifier_blindspot_indicator: int = Field(..., ge=0, le=1, description="beta_blind in {0, 1}")
    node_credit_weights: Dict[str, float] = Field(
        default_factory=dict, description="Softmax distribution alpha_v over DAG nodes."
    )

    @model_validator(mode="after")
    def validate_probabilities(self) -> "BackpropHint":
        if self.node_credit_weights:
            total = sum(self.node_credit_weights.values())
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"node_credit_weights must sum to 1.0, got {total}")
        return self


class SanitizedDiagnosticPayload(BaseModel):
    """Strict Inter-Process Telemetry Contract bridging Track A (Sandbox) and Track B (Meta-Architect)."""
    problem_id: str
    domain_archetype: str
    compute_tier: str
    public_pass_rate: float = Field(..., ge=0.0, le=1.0)
    gate_telemetry: GateTelemetry
    structural_failure: StructuralFailure
    dense_fitness: DenseFitnessVec
    backprop_hint: Optional[BackpropHint] = None
