from __future__ import annotations

import pytest

from moe_perf_model.demo import write_demo
from moe_perf_model.model import ComponentModel
from moe_perf_model.schema import load_training_records


@pytest.fixture(scope="session")
def demo_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    paths = write_demo(root)
    records = load_training_records(paths["training_records"])
    model = ComponentModel().fit(records, allow_synthetic=True)
    return root, paths, records, model

