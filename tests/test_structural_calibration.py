import pytest

from moe_perf_model.structural_calibration import (
    PhaseAffineCalibrator,
    StructuralCalibrationSample,
)


def test_phase_affine_recovers_phase_scales_and_runtime_offset():
    samples = []
    for workload, structural in enumerate((10.0, 20.0, 40.0)):
        samples.append(StructuralCalibrationSample(
            "prefill", structural, 1.5 * structural + 3.0, workload
        ))
        samples.append(StructuralCalibrationSample(
            "decode", structural, 2.0 * structural + 3.0, workload
        ))
    model = PhaseAffineCalibrator().fit(samples)
    assert model.alpha_by_phase["prefill"] == pytest.approx(1.5)
    assert model.alpha_by_phase["decode"] == pytest.approx(2.0)
    assert model.beta_runtime_us == pytest.approx(3.0)
    assert model.predict("prefill", 30.0) == pytest.approx(48.0)
    assert not model.provenance["total_forward_residual_model"]


def test_phase_affine_rejects_unknown_phase_and_underdetermined_fit():
    with pytest.raises(ValueError, match="more samples"):
        PhaseAffineCalibrator().fit([
            StructuralCalibrationSample("prefill", 1.0, 2.0, 0)
        ])
    model = PhaseAffineCalibrator().fit([
        StructuralCalibrationSample("prefill", 1.0, 2.0, 0),
        StructuralCalibrationSample("prefill", 2.0, 4.0, 1),
    ])
    with pytest.raises(KeyError):
        model.predict("decode", 1.0)
