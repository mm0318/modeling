from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResponsePrediction:
    value: float
    extrapolated: bool
    nearest_distance: float


def _finite_array(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


class IDWResponse:
    """Small-data response surface with normalized inverse-distance lookup."""

    def __init__(self, k: int = 4, power: float = 2.0):
        if k < 1 or power <= 0:
            raise ValueError("k and power must be positive")
        self.k = k
        self.power = power
        self.points: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, points: Iterable[Iterable[float]], values: Iterable[float]) -> "IDWResponse":
        points_array = _finite_array(points, "points")
        values_array = np.asarray(list(values), dtype=float)
        if values_array.ndim != 1 or len(values_array) != len(points_array):
            raise ValueError("values must be a vector aligned with points")
        if not np.isfinite(values_array).all() or np.any(values_array < 0):
            raise ValueError("response values must be finite and non-negative")

        grouped: dict[tuple[float, ...], list[float]] = {}
        for point, value in zip(points_array, values_array):
            grouped.setdefault(tuple(float(item) for item in point), []).append(float(value))
        self.points = np.asarray(list(grouped), dtype=float)
        self.values = np.asarray([median(items) for items in grouped.values()], dtype=float)
        self.minimum = np.min(self.points, axis=0)
        self.maximum = np.max(self.points, axis=0)
        self.scale = np.where(self.maximum > self.minimum, self.maximum - self.minimum, 1.0)
        return self

    def predict(self, point: Iterable[float]) -> ResponsePrediction:
        if self.points is None or self.values is None:
            raise RuntimeError("response surface has not been fitted")
        query = np.asarray(list(point), dtype=float)
        if query.shape != (self.points.shape[1],) or not np.isfinite(query).all():
            raise ValueError("query shape must match the fitted response dimensions")
        distances = np.linalg.norm((self.points - query) / self.scale, axis=1)
        nearest = float(np.min(distances))
        exact = np.flatnonzero(distances <= 1e-12)
        if len(exact):
            value = float(self.values[int(exact[0])])
        else:
            selected = np.argsort(distances)[: min(self.k, len(distances))]
            weights = 1.0 / np.maximum(distances[selected], 1e-12) ** self.power
            value = float(np.average(self.values[selected], weights=weights))
        extrapolated = bool(np.any(query < self.minimum) or np.any(query > self.maximum))
        return ResponsePrediction(max(0.0, value), extrapolated, nearest)


class PiecewiseLinearResponse:
    """Median-deduplicated 1D interpolation with explicit linear extrapolation."""

    def __init__(self):
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None

    def fit(self, keys: Iterable[float], values: Iterable[float]) -> "PiecewiseLinearResponse":
        grouped: dict[float, list[float]] = {}
        for key, value in zip(keys, values):
            key = float(key)
            value = float(value)
            if not math.isfinite(key) or not math.isfinite(value) or value < 0:
                raise ValueError("piecewise samples must be finite and values non-negative")
            grouped.setdefault(key, []).append(value)
        if not grouped:
            raise ValueError("piecewise response requires samples")
        pairs = sorted((key, median(items)) for key, items in grouped.items())
        self.keys = np.asarray([pair[0] for pair in pairs], dtype=float)
        self.values = np.asarray([pair[1] for pair in pairs], dtype=float)
        return self

    def predict(self, key: float) -> ResponsePrediction:
        if self.keys is None or self.values is None:
            raise RuntimeError("piecewise response has not been fitted")
        key = float(key)
        if not math.isfinite(key):
            raise ValueError("query key must be finite")
        extrapolated = key < self.keys[0] or key > self.keys[-1]
        if len(self.keys) == 1:
            return ResponsePrediction(float(self.values[0]), extrapolated, abs(key - self.keys[0]))
        if key <= self.keys[0]:
            lo, hi = 0, 1
        elif key >= self.keys[-1]:
            lo, hi = len(self.keys) - 2, len(self.keys) - 1
        else:
            hi = int(np.searchsorted(self.keys, key, side="right"))
            lo = hi - 1
        fraction = (key - self.keys[lo]) / (self.keys[hi] - self.keys[lo])
        value = self.values[lo] + fraction * (self.values[hi] - self.values[lo])
        nearest = float(np.min(np.abs(self.keys - key)) / max(np.ptp(self.keys), 1.0))
        return ResponsePrediction(max(0.0, float(value)), extrapolated, nearest)

