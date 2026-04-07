"""Tests for llama-perplexity output parsing."""

import pytest

from ppl_parse import parse_perplexity_output, format_perplexity_table, PerplexityResult


SAMPLE_OUTPUT_FULL = """\
main: seed = 42
llama_new_context_with_model: n_ctx_per_seq = 2048
system_info: n_threads = 8
perplexity: calculating perplexity over 655 chunks, batch_size=512
[1]14.8901,[2]10.1023,[3]8.5456,[4]7.8234,[5]7.2100,[6]6.9500,[7]6.7800,[8]6.5432

Final estimate: PPL = 6.5432 +/- 0.0321
"""

SAMPLE_OUTPUT_ALT_FORMAT = """\
perplexity = 8.2345 +/- 0.0678
"""

SAMPLE_OUTPUT_CHUNKS_ONLY = """\
[1]12.3456,[2]9.8765,[3]8.4321
"""


class TestParsePerplexityOutput:
    def test_full_output(self):
        r = parse_perplexity_output(SAMPLE_OUTPUT_FULL, "model.gguf")
        assert r.model_path == "model.gguf"
        assert r.final_ppl == 6.5432
        assert r.final_ppl_stderr == 0.0321
        assert r.n_chunks == 655
        assert len(r.chunk_ppls) == 8
        assert r.chunk_ppls[0] == 14.8901
        assert r.chunk_ppls[-1] == 6.5432

    def test_alt_format(self):
        r = parse_perplexity_output(SAMPLE_OUTPUT_ALT_FORMAT)
        assert r.final_ppl == 8.2345
        assert r.final_ppl_stderr == 0.0678

    def test_chunks_only_fallback(self):
        r = parse_perplexity_output(SAMPLE_OUTPUT_CHUNKS_ONLY)
        assert r.final_ppl == 8.4321  # last chunk value
        assert r.final_ppl_stderr == 0.0
        assert len(r.chunk_ppls) == 3
        assert r.n_chunks == 3

    def test_empty_output(self):
        r = parse_perplexity_output("")
        assert r.final_ppl == 0.0
        assert r.n_chunks == 0
        assert r.chunk_ppls == []

    def test_chunk_ordering(self):
        r = parse_perplexity_output(SAMPLE_OUTPUT_FULL)
        for i in range(1, len(r.chunk_ppls)):
            assert r.chunk_ppls[i] <= r.chunk_ppls[i - 1], (
                "Perplexity should decrease as more chunks are processed"
            )


class TestFormatPerplexityTable:
    def test_basic_table(self):
        results = [
            PerplexityResult("a.gguf", 6.5, 0.03, 100, []),
            PerplexityResult("b.gguf", 8.2, 0.05, 100, []),
        ]
        table = format_perplexity_table(results, ["Bonsai Q1_0", "Qwen3 Q4_K_M"])
        assert "Bonsai Q1_0" in table
        assert "6.5000" in table
        assert "8.2000" in table
        assert "Stderr" in table

    def test_table_has_header_and_separator(self):
        results = [PerplexityResult("a.gguf", 5.0, 0.01, 50, [])]
        table = format_perplexity_table(results, ["model1"])
        lines = table.strip().split("\n")
        assert len(lines) == 3  # header, separator, one data row
        assert lines[1].startswith("|---")
