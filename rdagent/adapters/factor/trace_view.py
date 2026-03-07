"""Compress a Trace object into a lightweight TraceView JSON for Planner/Evaluator."""
from __future__ import annotations

from typing import Any

from rdagent.core.proposal import Trace


def build_trace_view(
    trace: Trace,
    recent_rounds: int = 3,
    max_failed: int = 10,
) -> dict[str, Any]:
    """
    Convert a Trace into a compact JSON-serializable dict.

    Parameters
    ----------
    trace : Trace
        The full experiment trace.
    recent_rounds : int
        Number of recent rounds to include in detail.
    max_failed : int
        Max number of failed hypothesis summaries to include.

    Returns
    -------
    dict with keys: total_rounds, sota, recent_rounds, failed_hypotheses_summary
    """
    total = len(trace.hist)

    # --- SOTA ---
    sota = None
    for exp, feedback in reversed(trace.hist):
        if feedback.decision:
            round_id = trace.hist.index((exp, feedback))
            hyp_text = exp.hypothesis.hypothesis if exp.hypothesis else ""
            metrics = _extract_metrics(exp)
            sota = {
                "round_id": round_id,
                "hypothesis": hyp_text,
                **metrics,
            }
            break

    # --- Recent rounds ---
    recent = []
    start_idx = max(0, total - recent_rounds)
    for i in range(start_idx, total):
        exp, feedback = trace.hist[i]
        hyp_text = exp.hypothesis.hypothesis if exp.hypothesis else ""
        metrics = _extract_metrics(exp)
        observation = _build_observation(feedback, metrics)
        recent.append({
            "round_id": i,
            "hypothesis": hyp_text,
            "decision": feedback.decision,
            "key_metrics": metrics,
            "key_observation": observation,
        })

    # --- Failed hypotheses ---
    failed = []
    seen = set()
    for exp, feedback in trace.hist:
        if not feedback.decision and exp.hypothesis:
            reason = exp.hypothesis.concise_reason
            if reason and reason not in seen:
                seen.add(reason)
                failed.append(reason)
            if len(failed) >= max_failed:
                break

    return {
        "total_rounds": total,
        "sota": sota,
        "recent_rounds": recent,
        "failed_hypotheses_summary": failed,
    }


def _extract_metrics(exp) -> dict[str, Any]:
    """Extract numeric metrics from experiment result."""
    result = exp.result
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if isinstance(v, (int, float))}
    return {}


def _build_observation(feedback, metrics: dict) -> str:
    """Build a concise observation string from feedback."""
    parts = []
    if feedback.decision:
        parts.append("SOTA update")
    else:
        parts.append("No improvement")

    if "IC" in metrics:
        parts.append(f"IC={metrics['IC']:.4f}")

    if hasattr(feedback, "reason") and feedback.reason:
        # Truncate long reasons
        reason = feedback.reason
        if len(reason) > 100:
            reason = reason[:97] + "..."
        parts.append(reason)

    return "; ".join(parts)
