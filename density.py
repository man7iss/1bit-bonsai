"""Intelligence density calculator.

Paper's metric:  D = -log(P_e) / N
where P_e = 1 - (avg_score / 100), N = model size in GB.

Alternative:     D_ppl = 1 / (ppl * N)
Higher is better for both.
"""

import math
from dataclasses import dataclass


@dataclass
class ModelResult:
    name: str
    size_gb: float
    scores: dict  # benchmark_name -> score (0-100)
    perplexity: float | None = None


def intelligence(scores: dict) -> float:
    """Compute -log(1 - avg_score/100)."""
    avg = sum(scores.values()) / len(scores)
    if avg >= 100.0:
        raise ValueError(f"Average score {avg} >= 100; metric is undefined")
    if avg <= 0.0:
        raise ValueError(f"Average score {avg} <= 0; metric is undefined")
    return -math.log(1.0 - avg / 100.0)


def intelligence_density(scores: dict, size_gb: float) -> float:
    """Compute D = -log(P_e) / N. Units: 1/GB."""
    if size_gb <= 0:
        raise ValueError(f"Model size must be positive, got {size_gb}")
    return intelligence(scores) / size_gb


def perplexity_density(perplexity: float, size_gb: float) -> float:
    """Compute D_ppl = 1 / (ppl * N). Higher is better."""
    if perplexity <= 0:
        raise ValueError(f"Perplexity must be positive, got {perplexity}")
    if size_gb <= 0:
        raise ValueError(f"Model size must be positive, got {size_gb}")
    return 1.0 / (perplexity * size_gb)


def compute_density_table(results: list[ModelResult], benchmark_names: list[str] | None = None):
    """Compute intelligence density for a list of models, sorted descending."""
    rows = []
    for r in results:
        scores = r.scores
        if benchmark_names:
            scores = {k: v for k, v in scores.items() if k in benchmark_names}
        if not scores:
            continue

        avg_score = sum(scores.values()) / len(scores)
        density = intelligence_density(scores, r.size_gb)

        row = {
            "model": r.name,
            "size_gb": r.size_gb,
            "avg_score": round(avg_score, 2),
            "intelligence": round(intelligence(scores), 3),
            "density": round(density, 3),
        }
        if r.perplexity is not None:
            row["perplexity"] = r.perplexity
            row["ppl_density"] = round(perplexity_density(r.perplexity, r.size_gb), 6)

        rows.append(row)

    rows.sort(key=lambda r: r["density"], reverse=True)
    return rows


# Paper's reported results (Tables 7-9) for validation
PAPER_MODELS = [
    ModelResult("1-bit Bonsai 1.7B", 0.24, {
        "MMLU-Redux": 40.88, "GPQA Diamond": 43.2, "MuSR": 20.7, "IFEval": 45.1,
        "IFBench": 63.0, "GSM8K": 13.8, "MATH-500": 34.4, "HumanEval+": 45.1,
        "MBPP+": 42.3, "BFCLv3": 34.9,
    }),
    ModelResult("1-bit Bonsai 4B", 0.57, {
        "MMLU-Redux": 55.39, "GPQA Diamond": 58.7, "MuSR": 28.7, "IFEval": 41.4,
        "IFBench": 69.6, "GSM8K": 25.2, "MATH-500": 87.3, "HumanEval+": 71.3,
        "MBPP+": 57.9, "BFCLv3": 48.0,
    }),
    ModelResult("1-bit Bonsai 8B", 1.15, {
        "MMLU-Redux": 59.86, "GPQA Diamond": 65.7, "MuSR": 30.0, "IFEval": 50.0,
        "IFBench": 79.8, "GSM8K": 19.8, "MATH-500": 88.0, "HumanEval+": 73.8,
        "MBPP+": 59.8, "BFCLv3": 65.7,
    }),
    ModelResult("Qwen 3 8B", 16.38, {
        "MMLU-Redux": 71.02, "GPQA Diamond": 83.0, "MuSR": 49.3, "IFEval": 55.0,
        "IFBench": 81.5, "GSM8K": 27.2, "MATH-500": 93.0, "HumanEval+": 82.3,
        "MBPP+": 73.5, "BFCLv3": 81.0,
    }),
]

PAPER_MAIN_BENCHMARKS = [
    "MMLU-Redux", "MuSR", "GSM8K", "HumanEval+", "IFEval", "BFCLv3",
]
