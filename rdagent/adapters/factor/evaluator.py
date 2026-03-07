"""Evaluator subagent: generates feedback from run results via LLM."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from rdagent.adapters.artifact_utils import SCHEMA_VERSION, save_artifact
from rdagent.adapters.factor.trace_view import build_trace_view
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend as get_api_backend

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

EVALUATOR_SYSTEM_PROMPT = """\
You are a quantitative finance experiment evaluator. Your task is to assess \
whether a factor hypothesis was successful based on backtest metrics.

You evaluate ONLY based on metrics and hypothesis alignment — you do NOT see \
the implementation code. This separation ensures unbiased evaluation.

Respond ONLY with a JSON object matching the schema below. No extra text."""

EVALUATOR_USER_TEMPLATE = """\
## Hypothesis
{hypothesis_text}

## Factor Details
- Name: {factor_name}
- Description: {factor_description}
- Formulation: {factor_formulation}

## Backtest Results
Status: {status}
Metrics:
```json
{metrics}
```

## SOTA Baseline
{sota_section}

## Code Change Summary
{code_change_summary}

## Output Schema

Return a JSON object with these fields:

```json
{{
  "reason": "Detailed explanation of your evaluation",
  "decision": true/false,
  "code_change_summary": "Summary of what code changes were made",
  "observations": "Key observations from the metrics",
  "hypothesis_evaluation": "How well the hypothesis was supported",
  "new_hypothesis": "Suggestion for next round (or null if decision=true)",
  "acceptable": true/false
}}
```

Decision criteria:
- decision=true: IC improved over SOTA AND the factor shows consistent alpha
- decision=false: IC did not improve or results are unstable
- acceptable=true: The experiment ran correctly even if it didn't beat SOTA"""

REQUIRED_FIELDS = ["reason", "decision", "code_change_summary"]


def _validate_evaluator_output(data: dict) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    if "decision" in data and not isinstance(data["decision"], bool):
        errors.append(f"'decision' must be boolean, got {type(data['decision']).__name__}")
    return errors


def _build_sota_section(trace: Trace) -> str:
    """Build SOTA comparison section from trace."""
    sota_hyp, sota_exp = trace.get_sota_hypothesis_and_experiment()
    if sota_exp is None:
        return "No previous SOTA (this is the first experiment)."

    result = sota_exp.result
    if isinstance(result, dict):
        metrics_str = json.dumps(result, indent=2)
    else:
        metrics_str = str(result)

    hyp_text = sota_hyp.hypothesis if sota_hyp else "N/A"
    return f"SOTA Hypothesis: {hyp_text}\nSOTA Metrics:\n```json\n{metrics_str}\n```"


def _build_code_change_summary(exp: Experiment) -> str:
    """Extract code change summary from experiment feedback without exposing source code."""
    parts = []
    if exp.prop_dev_feedback is not None:
        fb = exp.prop_dev_feedback
        if hasattr(fb, "feedback_list"):
            for i, single_fb in enumerate(fb):
                if single_fb is not None:
                    if hasattr(single_fb, "execution") and single_fb.execution:
                        parts.append(f"Task {i}: {single_fb.execution}")
        elif hasattr(fb, "reason"):
            parts.append(str(fb))

    for ws in (exp.sub_workspace_list or []):
        if ws is not None and hasattr(ws, "change_summary") and ws.change_summary:
            parts.append(f"Change: {ws.change_summary}")

    return " | ".join(parts) if parts else "No change summary available."


class ClaudeCodeFactorEvaluatorAdapter(Experiment2Feedback):
    """
    LLM-driven experiment evaluator. Calls Claude via APIBackend,
    parses response into HypothesisFeedback, and writes feedback.json.
    """

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

        # Handle exception case without LLM
        if exception is not None:
            feedback = HypothesisFeedback(
                reason=f"Experiment failed with exception: {exception}",
                decision=False,
                code_change_summary="",
                acceptable=False,
            )
            self._write_feedback_artifact(feedback, round_idx, exception)
            return feedback

        # Build evaluator input
        hypothesis_text = exp.hypothesis.hypothesis if exp.hypothesis else "No hypothesis"
        factor_name = ""
        factor_description = ""
        factor_formulation = ""
        if exp.sub_tasks:
            task = exp.sub_tasks[0]
            factor_name = getattr(task, "factor_name", "")
            factor_description = getattr(task, "description", "")
            factor_formulation = getattr(task, "factor_formulation", "")

        result = exp.result
        status = "success"
        if result is None:
            status = "failed"
        metrics = json.dumps(result if isinstance(result, dict) else {"raw": str(result)}, indent=2)

        sota_section = _build_sota_section(trace)
        code_change_summary = _build_code_change_summary(exp)

        user_prompt = EVALUATOR_USER_TEMPLATE.format(
            hypothesis_text=hypothesis_text,
            factor_name=factor_name,
            factor_description=factor_description,
            factor_formulation=factor_formulation,
            status=status,
            metrics=metrics,
            sota_section=sota_section,
            code_change_summary=code_change_summary,
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
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                json_mode=True,
            )

            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                last_errors = [f"Response is not valid JSON: {response[:200]}"]
                logger.warning("Evaluator attempt %d: invalid JSON", attempt + 1)
                continue

            last_errors = _validate_evaluator_output(data)
            if not last_errors:
                break
            logger.warning("Evaluator attempt %d: validation errors: %s", attempt + 1, last_errors)

        if data is None or last_errors:
            if self.run_dir is not None:
                manifest_path = self.run_dir / f"round_{round_idx}" / "manifest.json"
                if manifest_path.exists():
                    from rdagent.adapters.artifact_utils import load_artifact
                    manifest = load_artifact(manifest_path)
                    manifest["failure_type"] = "schema_failure"
                    manifest["failure_errors"] = last_errors
                    save_artifact(manifest_path, manifest)
            raise RuntimeError(
                f"Evaluator failed after {MAX_RETRIES + 1} attempts. Errors: {last_errors}"
            )

        feedback = HypothesisFeedback(
            reason=data["reason"],
            decision=data["decision"],
            code_change_summary=data.get("code_change_summary", ""),
            observations=data.get("observations"),
            hypothesis_evaluation=data.get("hypothesis_evaluation"),
            new_hypothesis=data.get("new_hypothesis"),
            acceptable=data.get("acceptable"),
        )

        self._write_feedback_artifact(feedback, round_idx)
        return feedback

    def _write_feedback_artifact(
        self,
        feedback: HypothesisFeedback,
        round_idx: int,
        exception: Exception | None = None,
    ) -> None:
        if self.run_dir is None:
            return
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
