"""Rigid-body alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning mobile coordinates onto target coordinates."""

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float


def _validate_coordinate_array(coords: np.ndarray, name: str) -> np.ndarray:
    """Return a validated Nx3 coordinate array."""

    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); received {array.shape!r}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    return array


def compute_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    """Compute RMSD between coordinate arrays of identical shape."""

    ref = _validate_coordinate_array(reference, "reference")
    mob = _validate_coordinate_array(mobile, "mobile")
    if ref.shape != mob.shape:
        raise ValueError(
            f"reference and mobile must share the same shape; got {ref.shape!r} and {mob.shape!r}."
        )
    diff = ref - mob
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an Nx3 coordinate array."""

    xyz = _validate_coordinate_array(coords, "coords")
    rot = np.asarray(rotation, dtype=float)
    trans = np.asarray(translation, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3); received {rot.shape!r}.")
    if trans.shape != (3,):
        raise ValueError(f"translation must have shape (3,); received {trans.shape!r}.")
    return xyz @ rot + trans


def kabsch_superimpose(mobile: np.ndarray, target: np.ndarray) -> AlignmentResult:
    """Return the optimal rigid transform that maps mobile onto target."""

    mob = _validate_coordinate_array(mobile, "mobile")
    tgt = _validate_coordinate_array(target, "target")
    if mob.shape != tgt.shape:
        raise ValueError(
            f"mobile and target must share the same shape; got {mob.shape!r} and {tgt.shape!r}."
        )

    mobile_centroid = mob.mean(axis=0)
    target_centroid = tgt.mean(axis=0)
    mobile_centered = mob - mobile_centroid
    target_centered = tgt - target_centroid

    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    translation = target_centroid - mobile_centroid @ rotation
    aligned = apply_transform(mob, rotation, translation)

    return AlignmentResult(
        rotation=rotation,
        translation=translation,
        rmsd=compute_rmsd(tgt, aligned),
    )
