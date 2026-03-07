"""Planner subagent: generates hypothesis + experiment from TraceView via LLM."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.adapters.factor.trace_view import build_trace_view
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import ExperimentPlan
from rdagent.core.proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen, Trace
from rdagent.core.scenario import Scenario
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_utils import APIBackend as get_api_backend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# --- Prompt templates ---

PLANNER_SYSTEM_PROMPT = """\
You are a quantitative finance factor researcher. Your task is to propose a new \
alpha factor hypothesis and design an experiment to test it.

You will be given a TraceView summarizing past experiment rounds. Use it to:
1. Avoid repeating failed hypotheses.
2. Build on successful patterns.
3. Propose a novel, testable factor hypothesis.

Respond ONLY with a JSON object matching the schema below. No extra text."""

PLANNER_USER_TEMPLATE = """\
## TraceView (past experiments)
```json
{trace_view}
```

## Scenario
{scenario_desc}

## Output Schema

Return a single JSON object with exactly these fields:

```json
{{
  "hypothesis": "Full description of the factor hypothesis",
  "reason": "Detailed reasoning for why this factor should work",
  "concise_reason": "1-2 sentence summary of the reasoning",
  "concise_observation": "Key observation from data or literature",
  "concise_justification": "Why this hypothesis is justified",
  "concise_knowledge": "Relevant domain knowledge",
  "factor_name": "snake_case_factor_name",
  "factor_description": "Human-readable factor description",
  "factor_formulation": "Mathematical formula or pseudocode",
  "variables": {{"param1": "value1"}}
}}
```

IMPORTANT:
- Do NOT repeat hypotheses from failed_hypotheses_summary.
- factor_name must be a valid Python identifier.
- factor_formulation should be implementable in pandas."""

# --- Required fields for validation ---

REQUIRED_HYPOTHESIS_FIELDS = [
    "hypothesis", "reason", "concise_reason",
    "concise_observation", "concise_justification", "concise_knowledge",
]
REQUIRED_EXPERIMENT_FIELDS = [
    "factor_name", "factor_description", "factor_formulation", "variables",
]


def _validate_planner_output(data: dict) -> list[str]:
    """Validate planner output. Returns list of error messages (empty = valid)."""
    errors = []
    for field in REQUIRED_HYPOTHESIS_FIELDS + REQUIRED_EXPERIMENT_FIELDS:
        if field not in data or not data[field]:
            errors.append(f"Missing or empty required field: {field}")
    if "factor_name" in data and not data["factor_name"].isidentifier():
        errors.append(f"factor_name '{data.get('factor_name')}' is not a valid Python identifier")
    return errors


class ClaudeCodeFactorPlannerAdapter(HypothesisGen):
    """
    LLM-driven hypothesis generator. Calls Claude via APIBackend,
    parses response into Hypothesis, and writes hypothesis.json.
    """

    def __init__(self, scen: Scenario, run_dir: Path | None = None) -> None:
        super().__init__(scen)
        self.run_dir = run_dir
        self._call_count = 0

    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        round_idx = len(trace.hist)
        self._call_count += 1

        trace_view = build_trace_view(trace)
        scenario_desc = self.scen.get_scenario_all_desc(simple_background=True)

        user_prompt = PLANNER_USER_TEMPLATE.format(
            trace_view=json.dumps(trace_view, indent=2, ensure_ascii=False),
            scenario_desc=scenario_desc,
        )

        # Call LLM with retries
        backend = get_api_backend()
        data = None
        last_errors: list[str] = []

        for attempt in range(MAX_RETRIES + 1):
            retry_suffix = ""
            if attempt > 0 and last_errors:
                retry_suffix = (
                    f"\n\nYour previous response had validation errors:\n"
                    + "\n".join(f"- {e}" for e in last_errors)
                    + "\n\nPlease fix these and respond again."
                )

            response = backend.build_messages_and_create_chat_completion(
                user_prompt=user_prompt + retry_suffix,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                json_mode=True,
            )

            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                last_errors = [f"Response is not valid JSON: {response[:200]}"]
                logger.warning("Planner attempt %d: invalid JSON", attempt + 1)
                continue

            last_errors = _validate_planner_output(data)
            if not last_errors:
                break
            logger.warning("Planner attempt %d: validation errors: %s", attempt + 1, last_errors)

        if data is None or last_errors:
            # All retries failed — record schema_failure
            if self.run_dir is not None:
                manifest_path = self.run_dir / f"round_{round_idx}" / "manifest.json"
                if manifest_path.exists():
                    from rdagent.adapters.artifact_utils import load_artifact
                    manifest = load_artifact(manifest_path)
                    manifest["failure_type"] = "schema_failure"
                    manifest["failure_errors"] = last_errors
                    save_artifact(manifest_path, manifest)
            raise RuntimeError(
                f"Planner failed after {MAX_RETRIES + 1} attempts. Errors: {last_errors}"
            )

        # Build Hypothesis
        hypothesis = Hypothesis(
            hypothesis=data["hypothesis"],
            reason=data["reason"],
            concise_reason=data["concise_reason"],
            concise_observation=data["concise_observation"],
            concise_justification=data["concise_justification"],
            concise_knowledge=data["concise_knowledge"],
        )

        # Write hypothesis.json
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

        # Store experiment fields for H2E
        self._last_experiment_data = data
        return hypothesis


class ClaudeCodeFactorPlannerH2EAdapter(Hypothesis2Experiment):
    """
    Converts Planner output into a QlibFactorExperiment.
    Reads experiment fields from the planner's cached output.
    """

    def __init__(self, planner: ClaudeCodeFactorPlannerAdapter, run_dir: Path | None = None) -> None:
        self.planner = planner
        self.run_dir = run_dir
        self._call_count = 0

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> QlibFactorExperiment:
        round_idx = len(trace.hist)
        self._call_count += 1

        data = getattr(self.planner, "_last_experiment_data", None)
        if data is None:
            raise RuntimeError("No experiment data from planner. Call planner.gen() first.")

        task = FactorTask(
            factor_name=data["factor_name"],
            factor_description=data["factor_description"],
            factor_formulation=data["factor_formulation"],
            variables=data.get("variables", {}),
            factor_implementation=False,
        )

        # Build based_experiments
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

        # Write experiment.json
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
                        }
                    ],
                    "experiment_workspace": {
                        "class_path": "rdagent.scenarios.qlib.experiment.workspace.QlibFBWorkspace",
                        "workspace_path": str(exp.experiment_workspace.workspace_path),
                    },
                    "prop_dev_feedback": None,
                    "local_selection": None,
                },
            )

        return exp
