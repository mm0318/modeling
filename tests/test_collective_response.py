from __future__ import annotations

import pytest

from moe_perf_model.collective_model import CollectiveQuery, NcclResponseTable


def _table() -> NcclResponseTable:
    return NcclResponseTable({
        "schema_version": "nccl-response/v1",
        "profile_header": {
            "label_source": "nccl_benchmark",
            "formal_hardware_prediction": True,
            "topology": "single_node_nvlink",
        },
        "samples": [
            {"collective_type": "all_reduce", "rank_count": 2,
             "message_bytes_per_rank": 1024, "median_us": 10.0},
            {"collective_type": "all_reduce", "rank_count": 2,
             "message_bytes_per_rank": 4096, "median_us": 22.0},
        ],
    })


def test_nccl_response_interpolates_in_log_message_size():
    estimate = _table().predict(CollectiveQuery("all_reduce", 2048, 2, "intra_node", "nccl"))
    assert estimate.duration_us == pytest.approx(16.0)
    assert not estimate.provenance["extrapolated"]


def test_nccl_response_marks_range_extrapolation_and_rejects_topology():
    estimate = _table().predict(CollectiveQuery("all_reduce", 8192, 2, "intra_node", "nccl"))
    assert estimate.duration_us == 22.0
    assert estimate.provenance["extrapolated"]
    with pytest.raises(ValueError, match="single-node"):
        _table().predict(CollectiveQuery("all_reduce", 2048, 2, "inter_node", "nccl"))
