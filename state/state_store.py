import json
import sqlite3
from dataclasses import asdict
from typing import Dict, List, Optional

from harness.inner_harness import CognitiveHarness
from sandbox.runner import strict_pareto_dominates
from schemas.diagnostic_contract import BackpropHint, SanitizedDiagnosticPayload


class MAPElitesStateStore:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS issue_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                domain_archetype TEXT NOT NULL,
                failure_taxonomy TEXT NOT NULL,
                suspect_node TEXT NOT NULL,
                verifier_blindspot INTEGER NOT NULL,
                oracle_divergence REAL NOT NULL,
                sanitized_reason TEXT NOT NULL,
                resolved INTEGER DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS map_elites_archive (
                compute_tier TEXT NOT NULL,
                domain_archetype TEXT NOT NULL,
                harness_id TEXT NOT NULL,
                harness_json TEXT NOT NULL,
                vault_pass_rate REAL NOT NULL,
                normalized_tokens REAL NOT NULL,
                normalized_latency REAL NOT NULL,
                pareto_efficiency_score REAL NOT NULL,
                step_updated INTEGER NOT NULL,
                PRIMARY KEY (compute_tier, domain_archetype)
            )
            """
        )
        self.conn.commit()

    def log_issue(
        self, diagnostic: SanitizedDiagnosticPayload, hint: BackpropHint
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO issue_registry (
                problem_id, domain_archetype, failure_taxonomy,
                suspect_node, verifier_blindspot, oracle_divergence, sanitized_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                diagnostic.problem_id,
                diagnostic.domain_archetype,
                diagnostic.structural_failure.failure_taxonomy.value,
                hint.suspect_node.value,
                hint.verifier_blindspot_indicator,
                diagnostic.structural_failure.oracle_divergence,
                diagnostic.structural_failure.sanitized_reason,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def mark_domain_issues_resolved(self, domain_archetype: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE issue_registry SET resolved = 1 WHERE domain_archetype = ? AND resolved = 0",
            (domain_archetype,),
        )
        self.conn.commit()
        return int(cur.rowcount)

    def try_admit_candidate(
        self,
        harness: CognitiveHarness,
        diagnostic: SanitizedDiagnosticPayload,
        step_t: int,
    ) -> bool:
        c_tier = diagnostic.compute_tier
        domain = diagnostic.domain_archetype
        j_new = diagnostic.dense_fitness.objective_vector()

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT vault_pass_rate, normalized_tokens, normalized_latency
            FROM map_elites_archive
            WHERE compute_tier = ? AND domain_archetype = ?
            """,
            (c_tier, domain),
        )
        row = cur.fetchone()

        if row is not None:
            v_old, tau_old, mu_old = row
            j_old = [v_old, -tau_old, -mu_old]
            if not strict_pareto_dominates(j_new, j_old):
                return False

        cur.execute(
            """
            INSERT INTO map_elites_archive (
                compute_tier, domain_archetype, harness_id, harness_json,
                vault_pass_rate, normalized_tokens, normalized_latency,
                pareto_efficiency_score, step_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(compute_tier, domain_archetype) DO UPDATE SET
                harness_id=excluded.harness_id,
                harness_json=excluded.harness_json,
                vault_pass_rate=excluded.vault_pass_rate,
                normalized_tokens=excluded.normalized_tokens,
                normalized_latency=excluded.normalized_latency,
                pareto_efficiency_score=excluded.pareto_efficiency_score,
                step_updated=excluded.step_updated
            """,
            (
                c_tier,
                domain,
                harness.harness_id,
                json.dumps(asdict(harness)),
                diagnostic.dense_fitness.vault_pass_rate,
                diagnostic.dense_fitness.normalized_tokens,
                diagnostic.dense_fitness.normalized_latency,
                diagnostic.dense_fitness.pareto_efficiency_score,
                step_t,
            ),
        )
        self.conn.commit()
        return True

    def get_elite(self, compute_tier: str, domain_archetype: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT harness_id, harness_json, vault_pass_rate, normalized_tokens,
                   normalized_latency, pareto_efficiency_score, step_updated
            FROM map_elites_archive
            WHERE compute_tier = ? AND domain_archetype = ?
            """,
            (compute_tier, domain_archetype),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "harness_id": row[0],
            "harness": json.loads(row[1]),
            "vault_pass_rate": row[2],
            "normalized_tokens": row[3],
            "normalized_latency": row[4],
            "pareto_efficiency_score": row[5],
            "step_updated": row[6],
        }

    def list_all_elites(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT compute_tier, domain_archetype, harness_id, vault_pass_rate,
                   normalized_tokens, normalized_latency, pareto_efficiency_score, step_updated
            FROM map_elites_archive
            ORDER BY domain_archetype, compute_tier
            """
        )
        return [
            {
                "compute_tier": r[0],
                "domain_archetype": r[1],
                "harness_id": r[2],
                "vault_pass_rate": r[3],
                "normalized_tokens": r[4],
                "normalized_latency": r[5],
                "pareto_efficiency_score": r[6],
                "step_updated": r[7],
            }
            for r in cur.fetchall()
        ]

    def list_unresolved_issues(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, problem_id, domain_archetype, failure_taxonomy, suspect_node,
                   verifier_blindspot, oracle_divergence, sanitized_reason
            FROM issue_registry WHERE resolved = 0
            """
        )
        return [
            {
                "id": r[0],
                "problem_id": r[1],
                "domain_archetype": r[2],
                "failure_taxonomy": r[3],
                "suspect_node": r[4],
                "verifier_blindspot": r[5],
                "oracle_divergence": r[6],
                "sanitized_reason": r[7],
            }
            for r in cur.fetchall()
        ]
