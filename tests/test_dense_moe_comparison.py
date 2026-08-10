import json

from moe_perf_model.dense_moe_comparison import compare_dense_moe


def _row(family, workload, prompt, attention, ffn, step):
    timing = {"attention": attention, "step": step}
    timing["mlp" if family == "dense" else "expert"] = ffn
    return {
        "workload_index": workload,
        "repeat": 0,
        "step_index": 0,
        "phase": "prefill",
        "requested_workload": {
            "batch_size": 1,
            "prompt_tokens": prompt,
            "output_tokens": 2,
            "repeats": 1,
        },
        "requests": [{
            "phase": "prefill",
            "scheduled_tokens": prompt,
            "context_length": prompt,
        }],
        "timing_us": timing,
    }


def test_comparison_uses_common_ffn_definition_and_finds_transfer(tmp_path):
    dense_path = tmp_path / "dense.jsonl"
    moe_path = tmp_path / "moe.jsonl"
    dense_rows = [
        _row("dense", 0, 8, 20, 60, 100),
        _row("dense", 1, 8192, 70, 20, 100),
    ]
    moe_rows = [
        _row("moe", 0, 8, 20, 70, 100),
        _row("moe", 1, 8192, 60, 30, 100),
    ]
    dense_path.write_text("".join(json.dumps(row) + "\n" for row in dense_rows))
    moe_path.write_text("".join(json.dumps(row) + "\n" for row in moe_rows))

    report = compare_dense_moe(dense_path, moe_path)

    assert report["alignment"]["exact_aligned_steps"] == 2
    assert report["dense"]["transitions"][0]["from"] == "ffn"
    assert report["dense"]["transitions"][0]["to"] == "attention"
    assert report["moe"]["transitions"][0]["to"] == "attention"
    assert report["paired_point_count"] == 2
