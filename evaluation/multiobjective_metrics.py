"""Shared two-objective metrics for all molecular-generation baselines.

The project maximizes both EGFR and VEGFR2.  Every external baseline must write
canonical SMILES plus these two scores before using this module.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not len(points):
        return np.empty((0, 2), dtype=float)
    return points[nondominated_mask(points)]


def hypervolume_2d(points: np.ndarray, reference: Iterable[float] = (0.0, 0.0)) -> float:
    ref = np.asarray(tuple(reference), dtype=float)
    front = pareto_front(np.asarray(points, dtype=float))
    front = front[np.all(front > ref, axis=1)]
    if not len(front):
        return 0.0
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    best_y = ref[1]
    for x, y in front[::-1]:
        if y > best_y:
            hv += (x - ref[0]) * (y - best_y)
            best_y = y
    return float(hv)


def igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted generational distance plus for maximization objectives."""
    approx = np.asarray(approximation, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    if not len(approx) or not len(ref):
        return float("nan")
    distances = []
    for target in ref:
        # For maximization, only approximation deficits are penalized.
        deficits = np.maximum(target[None, :] - approx, 0.0)
        distances.append(np.linalg.norm(deficits, axis=1).min())
    return float(np.mean(distances))


def coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """Fraction of B weakly dominated by at least one point in A."""
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    if not len(b):
        return float("nan")
    if not len(a):
        return 0.0
    covered = [np.any(np.all(a >= point, axis=1)) for point in b]
    return float(np.mean(covered))


def spacing(front: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    if len(front) < 2:
        return float("nan")
    distances = []
    for i, point in enumerate(front):
        other = np.delete(front, i, axis=0)
        distances.append(np.abs(other - point).sum(axis=1).min())
    distances = np.asarray(distances)
    return float(np.sqrt(np.sum((distances - distances.mean()) ** 2) / (len(front) - 1)))


def spread(front: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    if not len(front):
        return float("nan")
    return float(np.linalg.norm(front.max(axis=0) - front.min(axis=0)))
