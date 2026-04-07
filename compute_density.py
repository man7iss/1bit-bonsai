"""
Compute and display intelligence density tables.

Combines paper results with our experimental measurements to produce
the comparison tables and validate the intelligence density metric.

Usage:
    python compute_density.py                    # Paper results only
    python compute_density.py --perplexity results/perplexity.csv
"""

import argparse
import csv
from pathlib import Path

from density import (
    ModelResult,
    compute_density_table,
    PAPER_MODELS,
    PAPER_MAIN_BENCHMARKS,
)


def load_perplexity_csv(path: str) -> dict[str, float]:
    """Load perplexity results from CSV. Returns {label: perplexity}."""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ppl = row.get("perplexity", "")
            if ppl and ppl != "N/A":
                results[row["model"]] = float(ppl)
    return results


def print_table(rows: list[dict], title: str = ""):
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    if not rows:
        print("(no data)")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compute intelligence density tables")
    parser.add_argument(
        "--perplexity", type=str, default=None,
        help="Path to perplexity CSV (from run_perplexity.sh)",
    )
    parser.add_argument(
        "--benchmarks", type=str, default=None, nargs="*",
        help="Benchmark names to use (default: paper's 6-benchmark main suite)",
    )
    args = parser.parse_args()

    benchmark_names = args.benchmarks or PAPER_MAIN_BENCHMARKS
    models = list(PAPER_MODELS)

    if args.perplexity and Path(args.perplexity).exists():
        ppl_data = load_perplexity_csv(args.perplexity)
        for m in models:
            if m.name in ppl_data:
                m.perplexity = ppl_data[m.name]

    # Full benchmark table (10 benchmarks)
    table_full = compute_density_table(models)
    print_table(table_full, "Intelligence Density (all 10 benchmarks)")

    # Main comparison (6 benchmarks, matching paper's Table 5)
    if benchmark_names != list(models[0].scores.keys()):
        table_main = compute_density_table(models, benchmark_names=benchmark_names)
        print_table(table_main, f"Intelligence Density ({len(benchmark_names)} benchmarks)")

    # Per-model breakdown
    print("\n\nPer-category deltas: Bonsai 8B vs Qwen 3 8B")
    print("=" * 50)
    bonsai = next(m for m in models if "Bonsai 8B" in m.name)
    qwen = next(m for m in models if "Qwen 3 8B" in m.name)
    deltas = []
    for bench in bonsai.scores:
        if bench in qwen.scores:
            delta = bonsai.scores[bench] - qwen.scores[bench]
            deltas.append({"benchmark": bench, "bonsai": bonsai.scores[bench],
                           "qwen": qwen.scores[bench], "delta": round(delta, 1)})
    deltas.sort(key=lambda x: x["delta"])
    print_table(deltas)


if __name__ == "__main__":
    main()
