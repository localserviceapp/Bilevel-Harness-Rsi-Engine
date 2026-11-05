import sys
import types
import numpy as np
from typing import Dict, Any, List, Tuple, Callable

try:
    import torch
except ImportError:
    torch = None


class AutonomousRSIDaemon:
    def __init__(self, swe_verifier, moe_router):
        self.swe_verifier = swe_verifier
        self.moe_router = moe_router

    def bisect_repository_contracts(
        self,
        repo: Dict[str, str],
        module_contracts: List[Tuple[str, Callable]],
        extra_modules: Dict[str, Any],
    ) -> Dict[str, int]:
        tracked = list(repo.keys()) + list(extra_modules.keys())
        saved = {k: sys.modules.get(k) for k in tracked}
        delta_slice = {}
        try:
            for k, m in extra_modules.items():
                sys.modules[k] = m
            env = {}
            for mod_name in repo:
                mod = types.ModuleType(mod_name)
                sys.modules[mod_name] = mod
                env[mod_name] = mod
            for mod_name, code_str in repo.items():
                exec(compile(code_str, f"<{mod_name}>", "exec"), env[mod_name].__dict__)
            for mod_name, pred in module_contracts:
                try:
                    ok = bool(pred(env))
                except Exception:
                    ok = False
                delta_slice[mod_name] = 0 if ok else 1
            return delta_slice
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

    @staticmethod
    def project_kway_orthogonal_subspace(prior_bases: List[Any], candidate_raw: Any) -> Tuple[Any, float]:
        if torch is not None and isinstance(candidate_raw, torch.Tensor):
            with torch.no_grad():
                joint = torch.cat([b.detach().to(torch.float64) for b in prior_bases], dim=1)
                cand_f = candidate_raw.detach().to(torch.float64)
                q_joint, _ = torch.linalg.qr(joint, mode="reduced")
                cand_orth = cand_f - q_joint @ (q_joint.T @ cand_f)
                cand_orth = cand_orth - q_joint @ (q_joint.T @ cand_orth)
                scale = torch.norm(prior_bases[0].detach().to(torch.float64), p="fro") / (
                    torch.norm(cand_orth, p="fro") + 1e-15
                )
                cand_orth = cand_orth * scale
                return cand_orth.to(candidate_raw.dtype), float(torch.norm(q_joint.T @ cand_orth, p="fro"))
        joint = np.concatenate([np.asarray(b, dtype=np.float64) for b in prior_bases], axis=1)
        cand_f = np.asarray(candidate_raw, dtype=np.float64)
        q_joint, _ = np.linalg.qr(joint, mode="reduced")
        cand_orth = cand_f - q_joint @ (q_joint.T @ cand_f)
        cand_orth = cand_orth - q_joint @ (q_joint.T @ cand_orth)
        scale = np.linalg.norm(np.asarray(prior_bases[0], dtype=np.float64), "fro") / (
            np.linalg.norm(cand_orth, "fro") + 1e-15
        )
        cand_orth = cand_orth * scale
        return cand_orth, float(np.linalg.norm(q_joint.T @ cand_orth, "fro"))
