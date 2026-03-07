"""Stub hypothesis generation adapter for Phase 1A testing."""
from __future__ import annotations

import uuid
from pathlib import Path

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.core.experiment import ExperimentPlan
from rdagent.core.proposal import Hypothesis, HypothesisGen, Trace
from rdagent.core.scenario import Scenario


class ClaudeCodeFactorHypothesisGenAdapter(HypothesisGen):
    """Stub: returns a fixed Hypothesis and writes hypothesis.json."""

    def __init__(self, scen: Scenario, run_dir: Path | None = None) -> None:
        super().__init__(scen)
        self.run_dir = run_dir
        self._call_count = 0

    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        round_idx = len(trace.hist)
        self._call_count += 1

        hypothesis = Hypothesis(
            hypothesis=f"Adding momentum factor improves alpha (round {round_idx + 1})",
            reason=f"Momentum is a well-known alpha factor. Round {round_idx + 1} test.",
            concise_reason="Momentum factor captures trend continuation",
            concise_observation="Price trends show persistence in short-term",
            concise_justification="Literature supports momentum premium across markets",
            concise_knowledge="Momentum factor = price return over lookback period",
        )

        if self.run_dir is not None:
            save_artifact(
                self.run_dir / f"round_{round_idx}" / "hypothesis.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "hypothesis",
                    "artifact_id": str(uuid.uuid4()),
                    "class_path": "rdagent.core.proposal.Hypothesis",
                    "hypothesis": hypothesis.hypothesis,
                    "reason": hypothesis.reason,
                    "concise_reason": hypothesis.concise_reason,
                    "concise_observation": hypothesis.concise_observation,
                    "concise_justification": hypothesis.concise_justification,
                    "concise_knowledge": hypothesis.concise_knowledge,
                },
            )

        return hypothesis
