from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class OperatorSample:
    operator_type: str
    phase: str
    numeric_shape: tuple[float, ...]
    dtype: str
    backend: str
    kernel_config: str
    latency_us: float

    @property
    def family(self) -> tuple[str, str, str, str, str]:
        return (self.operator_type, self.phase, self.dtype, self.backend, self.kernel_config)


@dataclass(frozen=True)
class OperatorEstimate:
    latency_us: float
    extrapolated: bool
    nearest_distance: float
    sample_count: int


class OperatorResponseDatabase:
    """Categorically strict, locally interpolated operator response database."""

    def __init__(self, neighbors: int = 4):
        self.neighbors = neighbors
        self.tables: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        self._exact_tables: dict[tuple[str, str, str, str, str], dict[tuple[float, ...], dict[str, Any]]] = {}
        self._query_cache: dict[tuple[Any, ...], OperatorEstimate] = {}

    def fit(self, samples: Iterable[OperatorSample]) -> "OperatorResponseDatabase":
        grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for sample in samples:
            if sample.latency_us < 0 or not math.isfinite(sample.latency_us):
                raise ValueError("operator latency must be non-negative and finite")
            grouped[(*sample.family, sample.numeric_shape)].append(sample.latency_us)
        families: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for key, values in grouped.items():
            family = key[:5]
            shape = tuple(float(value) for value in key[5])
            center = median(values)
            mad = median(abs(value - center) for value in values)
            families[family].append(
                {"shape": shape, "latency_us": center, "count": len(values), "mad_us": mad}
            )
        self.tables = {family: sorted(rows, key=lambda row: row["shape"]) for family, rows in families.items()}
        if not self.tables:
            raise ValueError("operator database requires at least one sample")
        self._rebuild_indexes()
        return self

    def _rebuild_indexes(self) -> None:
        self._exact_tables = {
            family: {tuple(row["shape"]): row for row in rows}
            for family, rows in self.tables.items()
        }
        self._query_cache = {}

    def _ensure_indexes(self) -> None:
        # Artifacts created before the index was introduced remain loadable.
        if not hasattr(self, "_exact_tables") or set(self._exact_tables) != set(self.tables):
            self._rebuild_indexes()
        elif not hasattr(self, "_query_cache"):
            self._query_cache = {}

    def query(
        self,
        operator_type: str,
        phase: str,
        numeric_shape: tuple[float, ...],
        dtype: str,
        backend: str,
        kernel_config: str,
    ) -> OperatorEstimate:
        family = (operator_type, phase, dtype, backend, kernel_config)
        rows = self.tables.get(family)
        if not rows:
            raise KeyError(f"missing operator response family: {family}")
        if any(len(row["shape"]) != len(numeric_shape) for row in rows):
            raise ValueError("operator numeric shape dimension mismatch")
        self._ensure_indexes()
        numeric_shape = tuple(float(value) for value in numeric_shape)
        cache_key = (*family, numeric_shape)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        row = self._exact_tables[family].get(numeric_shape)
        if row is not None:
            estimate = OperatorEstimate(row["latency_us"], False, 0.0, row["count"])
            self._query_cache[cache_key] = estimate
            return estimate
        dimensions = len(numeric_shape)
        mins = [min(row["shape"][index] for row in rows) for index in range(dimensions)]
        maxs = [max(row["shape"][index] for row in rows) for index in range(dimensions)]
        outside = any(value < mins[index] or value > maxs[index] for index, value in enumerate(numeric_shape))
        if dimensions == 1:
            ordered = rows
            x = numeric_shape[0]
            if x <= ordered[0]["shape"][0]:
                chosen = ordered[: min(2, len(ordered))]
            elif x >= ordered[-1]["shape"][0]:
                chosen = ordered[-min(2, len(ordered)) :]
            else:
                right = next(index for index, row in enumerate(ordered) if row["shape"][0] > x)
                chosen = ordered[right - 1 : right + 1]
            if len(chosen) == 1:
                latency = chosen[0]["latency_us"]
            else:
                x0, x1 = chosen[0]["shape"][0], chosen[1]["shape"][0]
                y0, y1 = chosen[0]["latency_us"], chosen[1]["latency_us"]
                latency = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            distance = min(abs(x - row["shape"][0]) / max(maxs[0] - mins[0], 1.0) for row in rows)
            estimate = OperatorEstimate(
                max(0.0, latency), outside, distance,
                sum(row["count"] for row in chosen),
            )
            self._query_cache[cache_key] = estimate
            return estimate
        spans = [max(maxs[index] - mins[index], 1.0) for index in range(dimensions)]
        distances = []
        for row in rows:
            distance = math.sqrt(
                sum(((numeric_shape[index] - row["shape"][index]) / spans[index]) ** 2 for index in range(dimensions))
            )
            distances.append((distance, row))
        chosen = sorted(distances, key=lambda item: item[0])[: self.neighbors]
        weights = [1.0 / max(distance, 1e-9) ** 2 for distance, _ in chosen]
        latency = sum(weight * row["latency_us"] for weight, (_, row) in zip(weights, chosen)) / sum(weights)
        estimate = OperatorEstimate(
            max(0.0, latency), outside, chosen[0][0], sum(row["count"] for _, row in chosen)
        )
        self._query_cache[cache_key] = estimate
        return estimate

    def stats(self) -> dict[str, int]:
        return {
            "families": len(self.tables),
            "unique_signatures": sum(len(rows) for rows in self.tables.values()),
            "raw_samples": sum(row["count"] for rows in self.tables.values() for row in rows),
        }

    def to_dict(self) -> dict[str, Any]:
        families = []
        for family, rows in sorted(self.tables.items()):
            operator_type, phase, dtype, backend, kernel_config = family
            families.append(
                {
                    "operator_type": operator_type,
                    "phase": phase,
                    "dtype": dtype,
                    "backend": backend,
                    "kernel_config": kernel_config,
                    "responses": rows,
                }
            )
        return {"stats": self.stats(), "families": families}
