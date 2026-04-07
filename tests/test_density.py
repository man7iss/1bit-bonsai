"""Tests for the intelligence density calculator.

Validates the metric implementation against the paper's reported values
(Table 6) and tests edge cases.
"""

import math
import pytest

from density import (
    intelligence,
    intelligence_density,
    perplexity_density,
    compute_density_table,
    ModelResult,
    PAPER_MODELS,
    PAPER_MAIN_BENCHMARKS,
)


class TestIntelligence:
    def test_perfect_score_raises(self):
        with pytest.raises(ValueError, match=">="):
            intelligence({"a": 100.0})

    def test_zero_score_raises(self):
        with pytest.raises(ValueError, match="<="):
            intelligence({"a": 0.0})

    def test_50_percent(self):
        result = intelligence({"a": 50.0})
        assert abs(result - math.log(2)) < 1e-10

    def test_90_percent(self):
        result = intelligence({"a": 90.0})
        expected = -math.log(0.1)
        assert abs(result - expected) < 1e-10

    def test_99_vs_90_roughly_2x(self):
        """Paper claims: a model scoring 99 is roughly 2x as intelligent as one scoring 90."""
        i99 = intelligence({"a": 99.0})
        i90 = intelligence({"a": 90.0})
        ratio = i99 / i90
        assert 1.9 < ratio < 2.1

    def test_55_vs_50_roughly_15pct(self):
        """Paper claims: a model scoring 55 is only ~15% more intelligent than one scoring 50."""
        i55 = intelligence({"a": 55.0})
        i50 = intelligence({"a": 50.0})
        pct_increase = (i55 - i50) / i50
        assert 0.10 < pct_increase < 0.20

    def test_multiple_benchmarks_averaged(self):
        scores = {"a": 80.0, "b": 60.0}
        result = intelligence(scores)
        expected = -math.log(1 - 70.0 / 100.0)
        assert abs(result - expected) < 1e-10


class TestIntelligenceDensity:
    def test_basic(self):
        scores = {"a": 70.0}
        density = intelligence_density(scores, 1.0)
        expected = -math.log(0.3)
        assert abs(density - expected) < 1e-10

    def test_smaller_model_higher_density(self):
        scores = {"a": 70.0}
        d_small = intelligence_density(scores, 1.0)
        d_large = intelligence_density(scores, 10.0)
        assert d_small > d_large
        assert abs(d_small / d_large - 10.0) < 1e-10

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            intelligence_density({"a": 50.0}, 0.0)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            intelligence_density({"a": 50.0}, -1.0)

    def test_paper_bonsai_8b_density(self):
        """Validate against paper's Table 6: Bonsai 8B density = 0.792 (10 benchmarks)."""
        bonsai = PAPER_MODELS[2]  # 1-bit Bonsai 8B
        density = intelligence_density(bonsai.scores, bonsai.size_gb)
        # Paper reports 0.792 for the 10-benchmark suite
        assert abs(density - 0.792) < 0.05, f"Expected ~0.792, got {density}"

    def test_paper_qwen3_8b_density(self):
        """Validate against paper's Table 6: Qwen 3 8B density = 0.076."""
        qwen = PAPER_MODELS[3]  # Qwen 3 8B
        density = intelligence_density(qwen.scores, qwen.size_gb)
        assert abs(density - 0.076) < 0.01, f"Expected ~0.076, got {density}"

    def test_bonsai_beats_qwen_on_density(self):
        """Core paper claim: Bonsai 8B has higher intelligence density than Qwen3 8B."""
        bonsai = PAPER_MODELS[2]
        qwen = PAPER_MODELS[3]
        d_bonsai = intelligence_density(bonsai.scores, bonsai.size_gb)
        d_qwen = intelligence_density(qwen.scores, qwen.size_gb)
        assert d_bonsai > d_qwen
        ratio = d_bonsai / d_qwen
        assert ratio > 8.0, f"Expected >8x density advantage, got {ratio}x"


class TestPerplexityDensity:
    def test_lower_perplexity_higher_density(self):
        d_good = perplexity_density(5.0, 1.0)
        d_bad = perplexity_density(10.0, 1.0)
        assert d_good > d_bad

    def test_smaller_model_higher_density(self):
        d_small = perplexity_density(5.0, 1.0)
        d_large = perplexity_density(5.0, 10.0)
        assert d_small > d_large

    def test_zero_perplexity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            perplexity_density(0.0, 1.0)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            perplexity_density(5.0, 0.0)


class TestComputeDensityTable:
    def test_sorts_by_density_descending(self):
        results = [
            ModelResult("big", 10.0, {"a": 70.0}),
            ModelResult("small", 1.0, {"a": 70.0}),
        ]
        table = compute_density_table(results)
        assert table[0]["model"] == "small"
        assert table[1]["model"] == "big"

    def test_filters_benchmarks(self):
        results = [
            ModelResult("m1", 1.0, {"a": 80.0, "b": 60.0, "c": 40.0}),
        ]
        table_all = compute_density_table(results)
        table_filtered = compute_density_table(results, benchmark_names=["a", "b"])
        assert table_all[0]["avg_score"] == 60.0
        assert table_filtered[0]["avg_score"] == 70.0

    def test_includes_perplexity_when_present(self):
        results = [
            ModelResult("m1", 1.0, {"a": 70.0}, perplexity=8.5),
        ]
        table = compute_density_table(results)
        assert "perplexity" in table[0]
        assert "ppl_density" in table[0]
        assert table[0]["perplexity"] == 8.5

    def test_excludes_perplexity_when_absent(self):
        results = [
            ModelResult("m1", 1.0, {"a": 70.0}),
        ]
        table = compute_density_table(results)
        assert "perplexity" not in table[0]

    def test_paper_models_full_table(self):
        """Validate full density table against paper's Table 6."""
        table = compute_density_table(PAPER_MODELS)
        names = [r["model"] for r in table]
        # Bonsai models should rank above Qwen3 on density
        bonsai_17b_idx = names.index("1-bit Bonsai 1.7B")
        bonsai_4b_idx = names.index("1-bit Bonsai 4B")
        bonsai_8b_idx = names.index("1-bit Bonsai 8B")
        qwen_idx = names.index("Qwen 3 8B")
        assert bonsai_17b_idx < qwen_idx
        assert bonsai_4b_idx < qwen_idx
        assert bonsai_8b_idx < qwen_idx

    def test_skips_models_with_no_matching_benchmarks(self):
        results = [
            ModelResult("m1", 1.0, {"a": 70.0}),
        ]
        table = compute_density_table(results, benchmark_names=["nonexistent"])
        assert len(table) == 0
