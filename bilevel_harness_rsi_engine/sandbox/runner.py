from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from sandbox.shadow_firewall import DefenseInDepthFirewall
from schemas.diagnostic_contract import (
    DenseFitnessVec,
    SanitizedDiagnosticPayload,
)


@dataclass
class ProblemInstance:
    problem_id: str
    domain_archetype: str
    specification: str
    func_name: str
    public_inputs: List[Any]
    shadow_inputs: List[Any]
    vault_inputs: List[Any]
    oracle_fn: Callable[[Any], Any]
    oracle_source: Optional[str] = None


def compute_pareto_efficiency_score(
    vault_pass_rate: float,
    normalized_tokens: float,
    normalized_latency: float,
    gamma_tau: float = 0.15,
    gamma_mu: float = 0.10,
) -> float:
    denom = ((1.0 + max(0.0, normalized_tokens)) ** gamma_tau) * (
        (1.0 + max(0.0, normalized_latency)) ** gamma_mu
    )
    return round(vault_pass_rate / denom, 6)


def strict_pareto_dominates(j_new: List[float], j_old: List[float]) -> bool:
    if len(j_new) != 3 or len(j_old) != 3:
        raise ValueError("Expected 3D objective vectors [V, -tau_bar, -mu_bar].")
    at_least_as_good = all(n >= o for n, o in zip(j_new, j_old))
    strictly_better = any(n > o for n, o in zip(j_new, j_old))
    return at_least_as_good and strictly_better


class ThreeTierShadowMembrane:
    def __init__(
        self,
        tau_0: float = 500.0,
        mu_0: float = 10.0,
        gamma_tau: float = 0.15,
        gamma_mu: float = 0.10,
    ):
        self.firewall = DefenseInDepthFirewall()
        self.tau_0 = tau_0
        self.mu_0 = mu_0
        self.gamma_tau = gamma_tau
        self.gamma_mu = gamma_mu

    def evaluate_candidate(
        self,
        code: str,
        problem: ProblemInstance,
        cumulative_tokens: int,
        compute_tier: str = "tier_standard",
    ) -> SanitizedDiagnosticPayload:
        ok_g0, g0_telemetry, g0_failure = self.firewall.gate_0_static_ast(code)
        tau_bar = round(cumulative_tokens / self.tau_0, 4)

        if not ok_g0:
            fitness = DenseFitnessVec(
                vault_pass_rate=0.0,
                normalized_tokens=tau_bar,
                normalized_latency=1.0,
                pareto_efficiency_score=0.0,
                invariants_passed_ratio=0.0,
                p95_latency_ms=0.0,
            )
            return SanitizedDiagnosticPayload(
                problem_id=problem.problem_id,
                domain_archetype=problem.domain_archetype,
                compute_tier=compute_tier,
                public_pass_rate=0.0,
                gate_telemetry=g0_telemetry,
                structural_failure=g0_failure,
                dense_fitness=fitness,
            )

        pub_outcomes, _, g_tel, s_fail = self.firewall.execute_in_isolated_sandbox(
            code, problem.func_name, problem.public_inputs, problem.oracle_fn, g0_telemetry
        )
        if g_tel.triggered_gate is not None:
            fitness = DenseFitnessVec(
                vault_pass_rate=0.0,
                normalized_tokens=tau_bar,
                normalized_latency=1.0,
                pareto_efficiency_score=0.0,
                invariants_passed_ratio=0.0,
                p95_latency_ms=0.0,
            )
            return SanitizedDiagnosticPayload(
                problem_id=problem.problem_id,
                domain_archetype=problem.domain_archetype,
                compute_tier=compute_tier,
                public_pass_rate=0.0,
                gate_telemetry=g_tel,
                structural_failure=s_fail,
                dense_fitness=fitness,
            )

        pub_pass_rate = round(sum(pub_outcomes) / max(1, len(pub_outcomes)), 4)

        sh_outcomes, sh_p95_ms, g_tel_sh, s_fail_sh = self.firewall.execute_in_isolated_sandbox(
            code, problem.func_name, problem.shadow_inputs, problem.oracle_fn, g_tel
        )
        sh_pass_rate = round(sum(sh_outcomes) / max(1, len(sh_outcomes)), 4)

        vault_outcomes, vault_p95_ms, _, _ = self.firewall.execute_in_isolated_sandbox(
            code, problem.func_name, problem.vault_inputs, problem.oracle_fn, g_tel
        )
        vault_pass_rate = round(sum(vault_outcomes) / max(1, len(vault_outcomes)), 4)

        p95_ms = round(max(sh_p95_ms, vault_p95_ms), 4)
        mu_bar = round(p95_ms / self.mu_0, 4)

        f_score = compute_pareto_efficiency_score(
            vault_pass_rate=vault_pass_rate,
            normalized_tokens=tau_bar,
            normalized_latency=mu_bar,
            gamma_tau=self.gamma_tau,
            gamma_mu=self.gamma_mu,
        )

        fitness = DenseFitnessVec(
            vault_pass_rate=vault_pass_rate,
            normalized_tokens=tau_bar,
            normalized_latency=mu_bar,
            pareto_efficiency_score=f_score,
            invariants_passed_ratio=sh_pass_rate,
            p95_latency_ms=p95_ms,
        )

        return SanitizedDiagnosticPayload(
            problem_id=problem.problem_id,
            domain_archetype=problem.domain_archetype,
            compute_tier=compute_tier,
            public_pass_rate=pub_pass_rate,
            gate_telemetry=g_tel_sh,
            structural_failure=s_fail_sh,
            dense_fitness=fitness,
        )
