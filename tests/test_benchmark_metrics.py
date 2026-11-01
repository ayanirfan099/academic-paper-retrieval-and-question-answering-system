import math
import runpy
from pathlib import Path

import pytest

metrics = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_scifact.py"))[
    "rank_metrics"
]


def test_retrieval_metrics_handle_misses_and_multiple_relevant_documents():
    assert metrics(["x", "y"], {"a"}) == {
        "recall_at_10": 0,
        "mrr_at_10": 0,
        "ndcg_at_10": 0,
    }
    result = metrics(["x", "a", "b"], {"a", "b"})
    assert result["recall_at_10"] == 1
    assert result["mrr_at_10"] == 0.5
    assert result["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))


def test_top_ten_cutoff_excludes_late_hits():
    assert metrics([str(i) for i in range(10)] + ["a"], {"a"})["recall_at_10"] == 0
