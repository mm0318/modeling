from moe_perf_model.communication_hooks import summarize_communication_events


def test_communication_summary_reports_union_overlap_and_owner():
    events = [
        {"group": "tp", "collective": "all_reduce", "method": "all_reduce",
         "message_bytes": 16, "start": 1.0, "end": 4.0},
        {"group": "ep", "collective": "all_gather", "method": "all_gatherv",
         "message_bytes": 32, "start": 3.0, "end": 6.0},
    ]
    report = summarize_communication_events(
        events,
        {"attention": [(0.0, 4.0)], "ffn": [(4.0, 8.0)]},
        lambda event: event,
        lambda start, end: end - start,
    )
    assert report["work_us"] == 6.0
    assert report["union_us"] == 5.0
    assert report["overlap_us"] == 1.0
    assert report["by_group_us"] == {"tp": 3.0, "ep": 3.0}
    assert report["by_component_us"] == {"attention": 3.0, "ffn": 3.0}
