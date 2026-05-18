from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    energy_mae: float
    energy_rmse: float
    force_component_mae: float
    force_component_rmse: float
    force_vector_mae: float
    force_vector_rmse: float
    force_vector_p95: float
    force_vector_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_same_shape(name: str, reference: np.ndarray, predicted: np.ndarray) -> None:
    if reference.shape != predicted.shape:
        raise ValueError(f"{name} arrays must have matching shapes: {reference.shape} != {predicted.shape}")


def summarize_errors(
    reference_energy: np.ndarray,
    predicted_energy: np.ndarray,
    reference_forces: np.ndarray,
    predicted_forces: np.ndarray,
) -> MetricSummary:
    reference_energy = np.asarray(reference_energy, dtype=float)
    predicted_energy = np.asarray(predicted_energy, dtype=float)
    reference_forces = np.asarray(reference_forces, dtype=float)
    predicted_forces = np.asarray(predicted_forces, dtype=float)

    _validate_same_shape("energy", reference_energy, predicted_energy)
    _validate_same_shape("force", reference_forces, predicted_forces)

    energy_error = predicted_energy - reference_energy
    force_error = predicted_forces - reference_forces
    force_vector_error = np.linalg.norm(force_error, axis=-1)

    return MetricSummary(
        energy_mae=float(np.mean(np.abs(energy_error))),
        energy_rmse=float(np.sqrt(np.mean(energy_error**2))),
        force_component_mae=float(np.mean(np.abs(force_error))),
        force_component_rmse=float(np.sqrt(np.mean(force_error**2))),
        force_vector_mae=float(np.mean(force_vector_error)),
        force_vector_rmse=float(np.sqrt(np.mean(force_vector_error**2))),
        force_vector_p95=float(np.percentile(force_vector_error, 95.0)),
        force_vector_max=float(np.max(force_vector_error)),
    )
