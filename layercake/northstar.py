"""Machine-checkable acceptance contracts for LayerCake scale candidates.

The deployment target is CPU-first: "mobile" is treated as a constrained
single-thread/non-GPU proxy and "desktop" as a broader CPU-class deployment
gate. GPU measurements are tracked separately as accelerator gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NorthStarMetrics:
    parameters: int
    baseline_parameters: int
    heldout_bpb: float
    baseline_heldout_bpb: float
    training_bytes: float
    baseline_training_bytes: float
    training_seconds: float
    baseline_training_seconds: float
    mobile_prefill_ratio: float
    mobile_generation_ratio: float
    desktop_prefill_ratio: float
    desktop_generation_ratio: float
    gpu_prefill_ratio: float
    gpu_generation_ratio: float
    migration_ppl_ratio: float
    migration_max_logit_diff: float
    migrated_domain_bpb: float
    baseline_domain_bpb: float

    def gates(self) -> dict[str, bool]:
        return {
            "smaller_model": self.parameters < self.baseline_parameters,
            "better_quality": self.heldout_bpb < self.baseline_heldout_bpb,
            "no_more_training_bytes": (
                self.training_bytes <= self.baseline_training_bytes
            ),
            "faster_training": (
                self.training_seconds < self.baseline_training_seconds
            ),
            "faster_mobile_cpu_prefill": self.mobile_prefill_ratio > 1.0,
            "faster_mobile_cpu_generation": (
                self.mobile_generation_ratio > 1.0
            ),
            "faster_desktop_cpu_prefill": self.desktop_prefill_ratio > 1.0,
            "faster_desktop_cpu_generation": (
                self.desktop_generation_ratio > 1.0
            ),
            "faster_gpu_prefill": self.gpu_prefill_ratio > 1.0,
            "faster_gpu_generation": self.gpu_generation_ratio > 1.0,
            "lossless_migration_ppl": self.migration_ppl_ratio == 1.0,
            "lossless_migration_logits": (
                self.migration_max_logit_diff == 0.0
            ),
            "better_migrated_domain_quality": (
                self.migrated_domain_bpb < self.baseline_domain_bpb
            ),
        }

    def certificate(self) -> dict:
        gates = self.gates()
        failed = [name for name, passed in gates.items() if not passed]
        return {
            "status": "PASS" if not failed else "FAIL",
            "required_gates": gates,
            "failed_required": failed,
            "metrics": asdict(self),
        }
