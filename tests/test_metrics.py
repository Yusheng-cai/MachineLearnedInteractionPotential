import numpy as np

from water_mlip_benchmark.metrics import MetricSummary, summarize_errors


def test_summarize_errors() -> None:
    reference_energy = np.array([1.0, 2.0, 3.0])
    predicted_energy = np.array([1.1, 1.8, 3.3])
    reference_forces = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])
    predicted_forces = np.array([[[0.1, 0.0, 0.0]], [[0.8, 0.0, 0.0]], [[0.0, 1.5, 0.0]]])

    summary = summarize_errors(reference_energy, predicted_energy, reference_forces, predicted_forces)

    assert isinstance(summary, MetricSummary)
    assert round(summary.energy_mae, 6) == 0.2
    assert round(summary.force_component_mae, 6) == round((0.1 + 0.2 + 0.5) / 9.0, 6)
    assert summary.force_vector_p95 > 0.0


def test_summarize_errors_rejects_shape_mismatch() -> None:
    try:
        summarize_errors(np.array([1.0]), np.array([1.0, 2.0]), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    except ValueError as exc:
        assert "energy arrays must have matching shapes" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch failure")
