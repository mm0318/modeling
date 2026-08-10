from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StructuralCalibrationSample:
    phase: str
    structural_us: float
    actual_us: float
    workload_index: int


class PhaseAffineCalibrator:
    """Small interpretable correction: alpha_phase * structural + beta_runtime."""

    def __init__(self):
        self.phases: tuple[str, ...] = ()
        self.alpha_by_phase: dict[str, float] = {}
        self.beta_runtime_us = 0.0
        self.provenance: dict[str, Any] = {}

    def fit(self, samples: list[StructuralCalibrationSample]) -> "PhaseAffineCalibrator":
        if not samples:
            raise ValueError("phase affine calibration requires samples")
        self.phases = tuple(sorted({sample.phase for sample in samples}))
        if len(samples) <= len(self.phases):
            raise ValueError("phase affine calibration needs more samples than phases")
        design = np.zeros((len(samples), len(self.phases) + 1), dtype=np.float64)
        target = np.asarray([sample.actual_us for sample in samples], dtype=np.float64)
        for row, sample in enumerate(samples):
            design[row, self.phases.index(sample.phase)] = sample.structural_us
            design[row, -1] = 1.0
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        alpha = coefficients[:-1]
        if np.any(~np.isfinite(coefficients)) or np.any(alpha <= 0):
            raise ValueError("phase affine calibration produced invalid coefficients")
        self.alpha_by_phase = {
            phase: float(value) for phase, value in zip(self.phases, alpha)
        }
        self.beta_runtime_us = float(coefficients[-1])
        self.provenance = {
            "artifact_type": "structural-phase-affine/v1",
            "formula": "alpha_phase * structural_us + beta_runtime_us",
            "training_sample_count": len(samples),
            "training_workload_count": len({sample.workload_index for sample in samples}),
            "phases": list(self.phases),
            "alpha_by_phase": self.alpha_by_phase,
            "beta_runtime_us": self.beta_runtime_us,
            "total_forward_residual_model": False,
        }
        return self

    def predict(self, phase: str, structural_us: float) -> float:
        if phase not in self.alpha_by_phase:
            raise KeyError(f"calibration has no phase: {phase}")
        return max(0.0, self.alpha_by_phase[phase] * structural_us + self.beta_runtime_us)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": list(self.phases),
            "alpha_by_phase": self.alpha_by_phase,
            "beta_runtime_us": self.beta_runtime_us,
            "provenance": self.provenance,
        }
