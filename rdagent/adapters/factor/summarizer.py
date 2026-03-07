"""Stub summarizer adapter for Phase 1A testing."""
from __future__ import annotations

import uuid
from pathlib import Path

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.core.scenario import Scenario


class ClaudeCodeFactorSummarizerAdapter(Experiment2Feedback):
    """Stub: Round 0 → decision=True (SOTA update), Round 1 → decision=False."""

    def __init__(self, scen: Scenario, run_dir: Path | None = None) -> None:
        super().__init__(scen)
        self.run_dir = run_dir
        self._call_count = 0

    def generate_feedback(
        self,
        exp: Experiment,
        trace: Trace,
        exception: Exception | None = None,
    ) -> HypothesisFeedback:
        round_idx = len(trace.hist)
        self._call_count += 1

        if exception is not None:
            feedback = HypothesisFeedback(
                reason=f"Experiment failed: {exception}",
                decision=False,
                code_change_summary="",
                acceptable=False,
            )
        else:
            decision = round_idx == 0
            feedback = HypothesisFeedback(
                reason=f"Round {round_idx + 1} evaluation. {'SOTA update' if decision else 'No improvement'}.",
                decision=decision,
                code_change_summary="Added momentum_20d factor implementation.",
                observations="IC analysis performed on validation set.",
                hypothesis_evaluation="Hypothesis partially supported by backtest results.",
                new_hypothesis="Consider adding volatility-adjusted momentum." if not decision else None,
                acceptable=True,
            )

        if self.run_dir is not None:
            save_artifact(
                self.run_dir / f"round_{round_idx}" / "feedback.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "feedback",
                    "artifact_id": str(uuid.uuid4()),
                    "class_path": "rdagent.core.proposal.HypothesisFeedback",
                    "reason": feedback.reason,
                    "decision": feedback.decision,
                    "code_change_summary": feedback.code_change_summary,
                    "observations": feedback.observations,
                    "hypothesis_evaluation": feedback.hypothesis_evaluation,
                    "new_hypothesis": feedback.new_hypothesis,
                    "acceptable": feedback.acceptable,
                    "exception": (
                        {"type": type(exception).__name__, "message": str(exception)}
                        if exception
                        else None
                    ),
                },
            )

        return feedback
