"""Parser for llama-perplexity output.

Example output:
    perplexity: calculating perplexity over 655 chunks, batch_size=512
    [1]4.8901,[2]5.1023,[3]5.0456,...
    Final estimate: PPL = 6.1234 +/- 0.0567
"""

import re
from dataclasses import dataclass


@dataclass
class PerplexityResult:
    model_path: str
    final_ppl: float
    final_ppl_stderr: float
    n_chunks: int
    chunk_ppls: list[float]


def parse_perplexity_output(text: str, model_path: str = "") -> PerplexityResult:
    """Parse llama-perplexity stdout/stderr into structured results."""
    chunk_ppls = [float(m.group(2)) for m in re.finditer(r"\[(\d+)\]([\d.]+)", text)]

    header = re.search(r"calculating perplexity over (\d+) chunks", text)
    n_chunks = int(header.group(1)) if header else len(chunk_ppls)

    final_ppl, final_stderr = 0.0, 0.0
    final = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    if final:
        final_ppl, final_stderr = float(final.group(1)), float(final.group(2))
    else:
        alt = re.search(r"perplexity\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
        if alt:
            final_ppl, final_stderr = float(alt.group(1)), float(alt.group(2))
        elif chunk_ppls:
            final_ppl = chunk_ppls[-1]

    return PerplexityResult(
        model_path=model_path,
        final_ppl=final_ppl,
        final_ppl_stderr=final_stderr,
        n_chunks=n_chunks,
        chunk_ppls=chunk_ppls,
    )


def format_perplexity_table(results: list[PerplexityResult], labels: list[str]) -> str:
    """Format perplexity results as a markdown table."""
    lines = [
        "| Model | Perplexity | Stderr |",
        "|-------|-----------|--------|",
    ]
    for label, r in zip(labels, results):
        lines.append(f"| {label} | {r.final_ppl:.4f} | {r.final_ppl_stderr:.4f} |")
    return "\n".join(lines)
