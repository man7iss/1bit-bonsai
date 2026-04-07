"""Tests for llama-bench output parsing."""

import pytest

from bench_parse import (
    BenchResult,
    parse_csv,
    parse_markdown,
    format_comparison_table,
    _parse_size,
    _parse_params,
    _parse_test_type,
    _parse_throughput,
)


SAMPLE_CSV = """\
build_commit,build_number,cuda,vulkan,kompute,metal,sycl,rpc,gpu_blas,blas,cpu_info,gpu_info,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,use_mmap,embeddings,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
abc123,1234,0,0,0,1,0,,1,0,Apple M1 Pro,Apple M1 Pro,bonsai-8b-q1_0.gguf,llama 8B Q1_0,1151820864,8190000000,2048,512,8,0x0,0,50,f16,f16,99,none,0,0,0,,1,0,512,0,2026-04-05T12:00:00Z,1066666666,5000000,480.0,2.5
abc123,1234,0,0,0,1,0,,1,0,Apple M1 Pro,Apple M1 Pro,bonsai-8b-q1_0.gguf,llama 8B Q1_0,1151820864,8190000000,2048,512,8,0x0,0,50,f16,f16,99,none,0,0,0,,1,0,0,128,2026-04-05T12:00:01Z,1523809523,3000000,84.0,0.3
"""

SAMPLE_MARKDOWN = """\
| model                  |       size |     params | backend | ngl | test    |              t/s |
| ---------------------- | ---------: | ---------: | ------- | --: | ------- | ---------------: |
| llama 8B Q1_0          |   1.07 GiB |     8.03 B | Metal   |  99 | pp512   |   480.00 ± 2.50  |
| llama 8B Q1_0          |   1.07 GiB |     8.03 B | Metal   |  99 | tg128   |    84.00 ± 0.30  |
| llama 8B Q4_K_M        |   4.58 GiB |     8.03 B | Metal   |  99 | pp512   |   450.00 ± 3.00  |
| llama 8B Q4_K_M        |   4.58 GiB |     8.03 B | Metal   |  99 | tg128   |    32.50 ± 0.20  |
"""


class TestParseHelpers:
    def test_parse_size_gib(self):
        assert _parse_size("1.07 GiB") == int(1.07 * 2**30)

    def test_parse_size_gb(self):
        assert _parse_size("1.15 GB") == int(1.15 * 1e9)

    def test_parse_size_mib(self):
        assert _parse_size("245 MiB") == int(245 * 2**20)

    def test_parse_size_invalid(self):
        assert _parse_size("unknown") == 0

    def test_parse_params_billions(self):
        assert _parse_params("8.03 B") == 8_030_000_000

    def test_parse_params_raw_int(self):
        assert _parse_params("8030000000") == 8_030_000_000

    def test_parse_params_invalid(self):
        assert _parse_params("unknown") == 0

    def test_parse_test_type_pp(self):
        assert _parse_test_type("pp512") == (512, 0)

    def test_parse_test_type_tg(self):
        assert _parse_test_type("tg128") == (0, 128)

    def test_parse_test_type_unknown(self):
        assert _parse_test_type("unknown") == (0, 0)

    def test_parse_throughput_with_stddev(self):
        avg, std = _parse_throughput("480.00 ± 2.50")
        assert avg == 480.0
        assert std == 2.5

    def test_parse_throughput_without_stddev(self):
        avg, std = _parse_throughput("480.0")
        assert avg == 480.0
        assert std == 0.0

    def test_parse_throughput_invalid(self):
        avg, std = _parse_throughput("N/A")
        assert avg == 0.0
        assert std == 0.0


class TestParseCSV:
    def test_parse_two_rows(self):
        results = parse_csv(SAMPLE_CSV)
        assert len(results) == 2

    def test_first_row_is_pp(self):
        results = parse_csv(SAMPLE_CSV)
        r = results[0]
        assert r.is_prompt_processing
        assert not r.is_token_generation
        assert r.n_prompt == 512
        assert r.n_gen == 0
        assert r.avg_ts == 480.0
        assert r.label == "pp512"

    def test_second_row_is_tg(self):
        results = parse_csv(SAMPLE_CSV)
        r = results[1]
        assert r.is_token_generation
        assert not r.is_prompt_processing
        assert r.n_prompt == 0
        assert r.n_gen == 128
        assert r.avg_ts == 84.0
        assert r.label == "tg128"

    def test_model_metadata(self):
        results = parse_csv(SAMPLE_CSV)
        r = results[0]
        assert r.model_filename == "bonsai-8b-q1_0.gguf"
        assert r.model_type == "llama 8B Q1_0"
        assert r.model_size_bytes == 1151820864
        assert abs(r.model_size_gb - 1.15182) < 0.001
        assert r.n_gpu_layers == 99


class TestParseMarkdown:
    def test_parse_four_rows(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        assert len(results) == 4

    def test_q1_pp(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        r = results[0]
        assert r.model_type == "llama 8B Q1_0"
        assert r.is_prompt_processing
        assert r.avg_ts == 480.0
        assert r.stddev_ts == 2.5

    def test_q4km_tg(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        r = results[3]
        assert r.model_type == "llama 8B Q4_K_M"
        assert r.is_token_generation
        assert r.avg_ts == 32.5

    def test_ngl_parsed(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        for r in results:
            assert r.n_gpu_layers == 99


class TestBenchResult:
    def test_model_size_gb(self):
        r = BenchResult("f", "t", 1_150_000_000, 8_000_000_000, 0, 128, 80.0, 1.0, 99, "")
        assert abs(r.model_size_gb - 1.15) < 0.001

    def test_label_pp(self):
        r = BenchResult("f", "t", 0, 0, 512, 0, 0, 0, 0, "")
        assert r.label == "pp512"

    def test_label_tg(self):
        r = BenchResult("f", "t", 0, 0, 0, 128, 0, 0, 0, "")
        assert r.label == "tg128"

    def test_label_mixed(self):
        r = BenchResult("f", "t", 0, 0, 512, 128, 0, 0, 0, "")
        assert r.label == "pp512+tg128"


class TestFormatComparisonTable:
    def test_basic_table(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        table = format_comparison_table(results)
        assert "llama 8B Q1_0" in table
        assert "llama 8B Q4_K_M" in table
        assert "480.0" in table
        assert "84.0" in table

    def test_speedup_column(self):
        results = parse_markdown(SAMPLE_MARKDOWN)
        table = format_comparison_table(results, reference_model="Q4_K_M")
        assert "Speedup" in table
        # Q1_0 tg128=84.0 / Q4_K_M tg128=32.5 = 2.6x
        assert "2.6x" in table
