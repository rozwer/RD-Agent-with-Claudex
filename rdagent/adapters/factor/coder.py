"""Stub coder adapter for Phase 1A testing."""
from __future__ import annotations

import re
from pathlib import Path

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
from rdagent.core.developer import Developer
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

STUB_FACTOR_CODE = """\
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    \"\"\"20-day momentum factor.\"\"\"
    close = df["close"]
    momentum = close / close.shift(20) - 1
    return momentum

if __name__ == "__main__":
    df = pd.read_hdf("source_data.h5")
    result = calculate_factor(df)
    result.to_hdf("result.h5", key="data")
"""


class ClaudeCodeFactorCoderAdapter(Developer):
    """Stub: injects fixed factor.py code into each sub_workspace."""

    def __init__(self, scen: Scenario, run_dir: Path | None = None) -> None:
        super().__init__(scen)
        self.run_dir = run_dir
        self._call_count = 0

    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        self._call_count += 1

        feedback_list = []
        for i, task in enumerate(exp.sub_tasks):
            ws = exp.sub_workspace_list[i]
            if ws is None:
                ws = FactorFBWorkspace(target_task=task)
                exp.sub_workspace_list[i] = ws

            ws.file_dict["factor.py"] = STUB_FACTOR_CODE
            ws.target_task = task

            feedback_list.append(
                CoSTEERSingleFeedback(
                    execution="Execution succeeded without error.",
                    return_checking="Output file found and valid.",
                    code=STUB_FACTOR_CODE,
                    final_decision=True,
                )
            )

        exp.prop_dev_feedback = CoSTEERMultiFeedback(feedback_list)

        if self.run_dir is not None:
            round_idx = 0
            if hasattr(exp, "hypothesis") and exp.hypothesis:
                match = re.search(r"round (\d+)", exp.hypothesis.hypothesis)
                if match:
                    round_idx = int(match.group(1)) - 1

            impl_dir = self.run_dir / f"round_{round_idx}" / "implementations"
            impl_dir.mkdir(parents=True, exist_ok=True)
            for i, _task in enumerate(exp.sub_tasks):
                (impl_dir / f"factor_{i + 1:03d}.py").write_text(STUB_FACTOR_CODE)

        return exp
