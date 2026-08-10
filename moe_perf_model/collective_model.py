from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CollectiveQuery:
    collective_type: str
    message_bytes: int
    rank_count: int
    topology: str
    backend: str = "synthetic"

    def __post_init__(self) -> None:
        if self.collective_type not in {
            "all_to_all",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
        }:
            raise ValueError("unsupported collective type")
        if self.message_bytes < 0 or self.rank_count < 2:
            raise ValueError("collective requires non-negative bytes and at least two ranks")


@dataclass(frozen=True)
class CollectiveEstimate:
    duration_us: float
    provenance: dict[str, object]


class CollectiveTimingBackend(Protocol):
    def predict(self, query: CollectiveQuery) -> CollectiveEstimate: ...


class SyntheticCollectiveTimingModel:
    """Execution-engine fixture; never a formal hardware communication model."""

    def __init__(self, *, startup_us: float = 5.0, bandwidth_gbps: float = 100.0):
        if startup_us < 0 or bandwidth_gbps <= 0:
            raise ValueError("synthetic collective parameters are invalid")
        self.startup_us = float(startup_us)
        self.bandwidth_gbps = float(bandwidth_gbps)

    def predict(self, query: CollectiveQuery) -> CollectiveEstimate:
        factors = {
            "all_to_all": (query.rank_count - 1) / query.rank_count,
            "all_reduce": 2.0 * (query.rank_count - 1) / query.rank_count,
            "all_gather": (query.rank_count - 1) / query.rank_count,
            "reduce_scatter": (query.rank_count - 1) / query.rank_count,
        }
        transfer_us = query.message_bytes * 8.0 / (self.bandwidth_gbps * 1e3)
        duration = self.startup_us + transfer_us * factors[query.collective_type]
        return CollectiveEstimate(
            duration_us=duration,
            provenance={
                "label_source": "synthetic_fixture",
                "model": "startup_plus_effective_bandwidth",
                "startup_us": self.startup_us,
                "bandwidth_gbps": self.bandwidth_gbps,
                "query": asdict(query),
                "formal_hardware_prediction": False,
            },
        )


class TopologyAwareCollectiveTimingModel:
    """Route collective queries to separately calibrated intra/inter-node models."""

    def __init__(self, intra_node: CollectiveTimingBackend, inter_node: CollectiveTimingBackend):
        self.intra_node = intra_node
        self.inter_node = inter_node

    def predict(self, query: CollectiveQuery) -> CollectiveEstimate:
        backend = self.intra_node if query.topology == "intra_node" else self.inter_node
        estimate = backend.predict(query)
        provenance = dict(estimate.provenance)
        provenance["topology_router"] = query.topology
        return CollectiveEstimate(estimate.duration_us, provenance)


class NcclResponseTable:
    """Formal phase-strict interpolation over a measured NCCL response artifact."""

    def __init__(self, raw: dict[str, object]):
        if raw.get("schema_version") != "nccl-response/v1":
            raise ValueError("unsupported NCCL response schema")
        header = raw.get("profile_header")
        samples = raw.get("samples")
        if not isinstance(header, dict) or header.get("label_source") != "nccl_benchmark":
            raise ValueError("NCCL table requires benchmark provenance")
        if not isinstance(samples, list) or not samples:
            raise ValueError("NCCL response table is empty")
        self.header = dict(header)
        self.rows: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("invalid NCCL sample")
            key = (str(sample["collective_type"]), int(sample["rank_count"]))
            self.rows.setdefault(key, []).append(
                (int(sample["message_bytes_per_rank"]), float(sample["median_us"]))
            )
        for values in self.rows.values():
            values.sort()

    @classmethod
    def load(cls, path: str | Path) -> "NcclResponseTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, query: CollectiveQuery) -> CollectiveEstimate:
        if query.backend != "nccl":
            raise ValueError("NCCL response table only accepts backend=nccl")
        if query.topology not in {"intra_node", "single_node_nvlink"}:
            raise ValueError("NCCL artifact is calibrated only for single-node NVLink")
        key = (query.collective_type, query.rank_count)
        anchors = self.rows.get(key)
        if anchors is None:
            raise ValueError(f"NCCL artifact has no anchors for {key}")
        target = max(1, query.message_bytes)
        extrapolated = target < anchors[0][0] or target > anchors[-1][0]
        if target <= anchors[0][0]:
            latency = anchors[0][1]
            distance = abs(math.log2(target / anchors[0][0]))
        elif target >= anchors[-1][0]:
            latency = anchors[-1][1]
            distance = abs(math.log2(target / anchors[-1][0]))
        else:
            left, right = next(
                (left, right) for left, right in zip(anchors, anchors[1:])
                if left[0] <= target <= right[0]
            )
            fraction = math.log(target / left[0]) / math.log(right[0] / left[0])
            latency = left[1] + fraction * (right[1] - left[1])
            distance = min(abs(math.log2(target / left[0])), abs(math.log2(right[0] / target)))
        return CollectiveEstimate(
            duration_us=latency,
            provenance={
                **self.header,
                "query": asdict(query),
                "interpolation": "piecewise_linear_log_message_bytes",
                "extrapolated": extrapolated,
                "nearest_anchor_log2_distance": distance,
            },
        )
