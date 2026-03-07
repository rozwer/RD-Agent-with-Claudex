"""Stub hypothesis-to-experiment adapter for Phase 1A testing."""
from __future__ import annotations

import uuid
from pathlib import Path

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.proposal import Hypothesis, Hypothesis2Experiment, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class ClaudeCodeFactorH2EAdapter(Hypothesis2Experiment):
    """Stub: converts hypothesis into a QlibFactorExperiment with 1 FactorTask."""

    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self._call_count = 0

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> QlibFactorExperiment:
        round_idx = len(trace.hist)
        self._call_count += 1

        task = FactorTask(
            factor_name="momentum_20d",
            factor_description="20-day momentum factor: close / close_20d_ago - 1",
            factor_formulation="momentum = close / shift(close, 20) - 1",
            variables={"lookback": 20},
            factor_implementation=False,
        )

        based_experiments: list[QlibFactorExperiment] = []
        if round_idx == 0:
            based_experiments.append(QlibFactorExperiment(sub_tasks=[]))
        else:
            for exp, feedback in reversed(trace.hist):
                if feedback.decision:
                    based_experiments.append(exp)
                    break

        exp = QlibFactorExperiment(
            sub_tasks=[task],
            based_experiments=based_experiments,
            hypothesis=hypothesis,
        )

        ws = FactorFBWorkspace(target_task=task)
        exp.sub_workspace_list = [ws]

        if self.run_dir is not None:
            save_artifact(
                self.run_dir / f"round_{round_idx}" / "experiment.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "experiment",
                    "artifact_id": str(uuid.uuid4()),
                    "class_path": "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorExperiment",
                    "hypothesis_ref": f"round_{round_idx}/hypothesis.json",
                    "based_experiments": (
                        [f"round_{i}/experiment.json" for i in range(round_idx)]
                        if round_idx > 0
                        else ["baseline"]
                    ),
                    "sub_tasks": [
                        {
                            "class_path": "rdagent.components.coder.factor_coder.factor.FactorTask",
                            "factor_name": task.factor_name,
                            "factor_description": task.factor_description,
                            "factor_formulation": task.factor_formulation,
                            "variables": task.variables,
                            "version": task.version,
                            "base_code": task.base_code,
                            "factor_resources": task.factor_resources,
                            "factor_implementation": task.factor_implementation,
                        }
                    ],
                    "sub_workspace_list": [
                        {
                            "class_path": "rdagent.components.coder.factor_coder.factor.FactorFBWorkspace",
                            "task_index": 0,
                            "workspace_path": str(ws.workspace_path),
                            "file_dict": ws.file_dict,
                            "change_summary": ws.change_summary,
                            "raise_exception": ws.raise_exception,
                        }
                    ],
                    "experiment_workspace": {
                        "class_path": "rdagent.scenarios.qlib.experiment.workspace.QlibFBWorkspace",
                        "workspace_path": str(exp.experiment_workspace.workspace_path),
                        "file_dict": {},
                        "init_kwargs": {},
                    },
                    "prop_dev_feedback": None,
                    "local_selection": None,
                },
            )

        return exp
