"""Tests for Planner and Evaluator with mocked LLM (Tasks 1B.1-1B.2)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rdagent.adapters.artifact_utils import create_round_dir, create_run_dir
from rdagent.adapters.factor.evaluator import (
    ClaudeCodeFactorEvaluatorAdapter,
    _build_code_change_summary,
    _build_sota_section,
    _validate_evaluator_output,
)
from rdagent.adapters.factor.planner import (
    ClaudeCodeFactorPlannerAdapter,
    ClaudeCodeFactorPlannerH2EAdapter,
    _validate_planner_output,
)
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace


class MockScenario:
    @property
    def background(self):
        return "Qlib factor research"

    def get_source_data_desc(self, task=None):
        return "Market data"

    @property
    def rich_style_description(self):
        return "test"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None):
        return "Background of the scenario:\nQlib factor research"


VALID_PLANNER_RESPONSE = json.dumps({
    "hypothesis": "Volume-weighted momentum captures institutional flow",
    "reason": "Institutions trade with volume. Volume-weighted price momentum isolates informed trades.",
    "concise_reason": "Volume-weighted momentum isolates institutional flow signals",
    "concise_observation": "High volume bars predict short-term direction",
    "concise_justification": "Academic research supports volume-price momentum linkage",
    "concise_knowledge": "VWAP momentum = vwap / vwap.shift(20) - 1",
    "factor_name": "vwap_momentum_20d",
    "factor_description": "20-day VWAP momentum factor",
    "factor_formulation": "vwap / vwap.shift(20) - 1",
    "variables": {"lookback": 20},
})

VALID_EVALUATOR_RESPONSE = json.dumps({
    "reason": "IC improved from baseline. Factor shows consistent positive alpha.",
    "decision": True,
    "code_change_summary": "Implemented VWAP momentum with 20-day lookback",
    "observations": "IC=0.045, stable across validation windows",
    "hypothesis_evaluation": "Hypothesis supported by positive IC",
    "new_hypothesis": None,
    "acceptable": True,
})


# --- Validation tests ---

class TestPlannerValidation:
    def test_valid_output(self):
        data = json.loads(VALID_PLANNER_RESPONSE)
        assert _validate_planner_output(data) == []

    def test_missing_field(self):
        data = {"hypothesis": "test"}
        errors = _validate_planner_output(data)
        assert len(errors) > 0
        assert any("reason" in e for e in errors)

    def test_invalid_factor_name(self):
        data = json.loads(VALID_PLANNER_RESPONSE)
        data["factor_name"] = "invalid-name"
        errors = _validate_planner_output(data)
        assert any("identifier" in e for e in errors)


class TestEvaluatorValidation:
    def test_valid_output(self):
        data = json.loads(VALID_EVALUATOR_RESPONSE)
        assert _validate_evaluator_output(data) == []

    def test_missing_decision(self):
        data = {"reason": "test", "code_change_summary": "test"}
        errors = _validate_evaluator_output(data)
        assert any("decision" in e for e in errors)

    def test_non_bool_decision(self):
        data = {"reason": "test", "decision": "yes", "code_change_summary": "test"}
        errors = _validate_evaluator_output(data)
        assert any("boolean" in e for e in errors)


# --- Planner integration with mock ---

class TestPlannerWithMock:
    @patch("rdagent.adapters.factor.planner.get_api_backend")
    def test_planner_generates_hypothesis(self, mock_get_backend, tmp_path):
        mock_backend = MagicMock()
        mock_backend.build_messages_and_create_chat_completion.return_value = VALID_PLANNER_RESPONSE
        mock_get_backend.return_value = mock_backend

        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)

        trace = Trace(scen=scen)
        planner = ClaudeCodeFactorPlannerAdapter(scen, run_dir=run_dir)
        hypothesis = planner.gen(trace)

        assert hypothesis.hypothesis == "Volume-weighted momentum captures institutional flow"
        assert hypothesis.concise_reason != ""
        assert (run_dir / "round_0" / "hypothesis.json").exists()
        mock_backend.build_messages_and_create_chat_completion.assert_called_once()

    @patch("rdagent.adapters.factor.planner.get_api_backend")
    def test_planner_h2e_creates_experiment(self, mock_get_backend, tmp_path):
        mock_backend = MagicMock()
        mock_backend.build_messages_and_create_chat_completion.return_value = VALID_PLANNER_RESPONSE
        mock_get_backend.return_value = mock_backend

        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        planner = ClaudeCodeFactorPlannerAdapter(scen, run_dir=run_dir)
        h2e = ClaudeCodeFactorPlannerH2EAdapter(planner, run_dir=run_dir)

        hypothesis = planner.gen(trace)
        exp = h2e.convert(hypothesis, trace)

        assert len(exp.sub_tasks) == 1
        assert exp.sub_tasks[0].factor_name == "vwap_momentum_20d"
        assert len(exp.based_experiments) >= 1
        assert (run_dir / "round_0" / "experiment.json").exists()

    @patch("rdagent.adapters.factor.planner.get_api_backend")
    def test_planner_retries_on_invalid_json(self, mock_get_backend, tmp_path):
        mock_backend = MagicMock()
        mock_backend.build_messages_and_create_chat_completion.side_effect = [
            "not valid json",
            VALID_PLANNER_RESPONSE,
        ]
        mock_get_backend.return_value = mock_backend

        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        planner = ClaudeCodeFactorPlannerAdapter(scen, run_dir=run_dir)
        hypothesis = planner.gen(trace)

        assert hypothesis.hypothesis == "Volume-weighted momentum captures institutional flow"
        assert mock_backend.build_messages_and_create_chat_completion.call_count == 2

    @patch("rdagent.adapters.factor.planner.get_api_backend")
    def test_planner_fails_after_max_retries(self, mock_get_backend, tmp_path):
        mock_backend = MagicMock()
        mock_backend.build_messages_and_create_chat_completion.return_value = "invalid"
        mock_get_backend.return_value = mock_backend

        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        planner = ClaudeCodeFactorPlannerAdapter(scen, run_dir=run_dir)
        with pytest.raises(RuntimeError, match="Planner failed"):
            planner.gen(trace)


# --- Evaluator integration with mock ---

class MockExperiment:
    def __init__(self, hypothesis, result=None):
        self.hypothesis = hypothesis
        self.result = result
        self.sub_tasks = []
        self.sub_workspace_list = []
        self.based_experiments = []
        self.experiment_workspace = None
        self.prop_dev_feedback = None
        self.running_info = type("RI", (), {"result": result})()


class TestEvaluatorWithMock:
    @patch("rdagent.adapters.factor.evaluator.get_api_backend")
    def test_evaluator_generates_feedback(self, mock_get_backend, tmp_path):
        mock_backend = MagicMock()
        mock_backend.build_messages_and_create_chat_completion.return_value = VALID_EVALUATOR_RESPONSE
        mock_get_backend.return_value = mock_backend

        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        hyp = Hypothesis(
            hypothesis="test", reason="test", concise_reason="test",
            concise_observation="test", concise_justification="test", concise_knowledge="test",
        )
        exp = MockExperiment(hyp, result={"IC": 0.045})

        evaluator = ClaudeCodeFactorEvaluatorAdapter(scen, run_dir=run_dir)
        feedback = evaluator.generate_feedback(exp, trace)

        assert feedback.decision is True
        assert "IC improved" in feedback.reason
        assert (run_dir / "round_0" / "feedback.json").exists()

    @patch("rdagent.adapters.factor.evaluator.get_api_backend")
    def test_evaluator_handles_exception(self, mock_get_backend, tmp_path):
        scen = MockScenario()
        run_dir = create_run_dir(tmp_path, "test_run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        hyp = Hypothesis(
            hypothesis="test", reason="test", concise_reason="test",
            concise_observation="test", concise_justification="test", concise_knowledge="test",
        )
        exp = MockExperiment(hyp)

        evaluator = ClaudeCodeFactorEvaluatorAdapter(scen, run_dir=run_dir)
        feedback = evaluator.generate_feedback(exp, trace, exception=RuntimeError("timeout"))

        assert feedback.decision is False
        assert "exception" in feedback.reason.lower() or "failed" in feedback.reason.lower()

    def test_information_separation(self):
        """Verify evaluator prompt does NOT contain source code."""
        from rdagent.adapters.factor.evaluator import EVALUATOR_USER_TEMPLATE
        assert "factor.py" not in EVALUATOR_USER_TEMPLATE
        assert "import pandas" not in EVALUATOR_USER_TEMPLATE
