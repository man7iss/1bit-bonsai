"""Parser for llama-bench CSV and markdown output."""

import csv
import io
import re
from dataclasses import dataclass

_SIZE_MULTIPLIERS = {"GB": 1e9, "GiB": 2**30, "MB": 1e6, "MiB": 2**20}


@dataclass
class BenchResult:
    model_filename: str
    model_type: str
    model_size_bytes: int
    model_n_params: int
    n_prompt: int
    n_gen: int
    avg_ts: float
    stddev_ts: float
    n_gpu_layers: int
    test_time: str

    @property
    def model_size_gb(self) -> float:
        return self.model_size_bytes / 1e9

    @property
    def is_prompt_processing(self) -> bool:
        return self.n_gen == 0

    @property
    def is_token_generation(self) -> bool:
        return self.n_prompt == 0

    @property
    def label(self) -> str:
        if self.n_gen == 0:
            return f"pp{self.n_prompt}"
        if self.n_prompt == 0:
            return f"tg{self.n_gen}"
        return f"pp{self.n_prompt}+tg{self.n_gen}"


def parse_csv(text: str) -> list[BenchResult]:
    """Parse llama-bench CSV output (produced with -o csv)."""
    return [
        BenchResult(
            model_filename=row["model_filename"],
            model_type=row.get("model_type", ""),
            model_size_bytes=int(row["model_size"]),
            model_n_params=int(row["model_n_params"]),
            n_prompt=int(row["n_prompt"]),
            n_gen=int(row["n_gen"]),
            avg_ts=float(row["avg_ts"]),
            stddev_ts=float(row["stddev_ts"]),
            n_gpu_layers=int(row["n_gpu_layers"]),
            test_time=row.get("test_time", ""),
        )
        for row in csv.DictReader(io.StringIO(text))
    ]


def parse_markdown(text: str) -> list[BenchResult]:
    """Parse llama-bench markdown table output (default format)."""
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|") or "| model" in line.lower():
            continue
        if not line.replace("|", "").replace("-", "").replace(":", "").strip():
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 7:
            continue

        n_prompt, n_gen = _parse_test_type(parts[5])
        avg_ts, stddev_ts = _parse_throughput(parts[6])

        results.append(BenchResult(
            model_filename="",
            model_type=parts[0],
            model_size_bytes=_parse_size(parts[1]),
            model_n_params=_parse_params(parts[2]),
            n_prompt=n_prompt,
            n_gen=n_gen,
            avg_ts=avg_ts,
            stddev_ts=stddev_ts,
            n_gpu_layers=int(parts[4]) if parts[4].strip().isdigit() else 0,
            test_time="",
        ))

    return results


def _parse_size(s: str) -> int:
    m = re.match(r"([\d.]+)\s*(GiB|GB|MiB|MB)", s.strip())
    if not m:
        return 0
    return int(float(m.group(1)) * _SIZE_MULTIPLIERS[m.group(2)])


def _parse_params(s: str) -> int:
    s = s.strip()
    m = re.match(r"([\d.]+)\s*B", s)
    if m:
        return round(float(m.group(1)) * 1e9)
    return int(s) if s.isdigit() else 0


def _parse_test_type(s: str) -> tuple[int, int]:
    m = re.match(r"(pp|tg)(\d+)", s.strip())
    if not m:
        return 0, 0
    n = int(m.group(2))
    return (n, 0) if m.group(1) == "pp" else (0, n)


def _parse_throughput(s: str) -> tuple[float, float]:
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", s.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s.strip()), 0.0
    except ValueError:
        return 0.0, 0.0


def format_comparison_table(results: list[BenchResult], reference_model: str = "") -> str:
    """Format results as a markdown comparison table."""
    ref_tg = None
    if reference_model:
        ref_tg = next(
            (r.avg_ts for r in results
             if reference_model in r.model_type and r.is_token_generation),
            None,
        )

    header = "| Model | Size | pp512 (tok/s) | tg128 (tok/s) |"
    sep = "|-------|------|---------------|---------------|"
    if ref_tg:
        header += " Speedup |"
        sep += "---------|"

    models = {}
    for r in results:
        key = r.model_type or r.model_filename
        if key not in models:
            models[key] = {"pp": None, "tg": None, "size": r.model_size_gb}
        if r.is_prompt_processing:
            models[key]["pp"] = r
        elif r.is_token_generation:
            models[key]["tg"] = r

    lines = [header, sep]
    for name, data in models.items():
        pp_str = f"{data['pp'].avg_ts:.1f}" if data["pp"] else "-"
        tg_str = f"{data['tg'].avg_ts:.1f}" if data["tg"] else "-"
        row = f"| {name} | {data['size']:.2f} GB | {pp_str} | {tg_str} |"
        if ref_tg and data["tg"]:
            row += f" {data['tg'].avg_ts / ref_tg:.1f}x |"
        elif ref_tg:
            row += " - |"
        lines.append(row)

    return "\n".join(lines)
