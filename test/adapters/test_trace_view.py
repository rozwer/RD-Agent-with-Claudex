"""Tests for TraceView generation (Task 1B.1.1)."""
import pytest

from rdagent.adapters.factor.trace_view import build_trace_view
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace


class MockScenario:
    @property
    def background(self):
        return "test"

    def get_source_data_desc(self, task=None):
        return "test"

    @property
    def rich_style_description(self):
        return "test"


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
        self.sub_results = {}
        self.local_selection = None
        self.plan = None
        self.user_instructions = None


class TestBuildTraceViewEmpty:
    def test_empty_trace(self):
        scen = MockScenario()
        trace = Trace(scen=scen)
        view = build_trace_view(trace)

        assert view["total_rounds"] == 0
        assert view["sota"] is None
        assert view["recent_rounds"] == []
        assert view["failed_hypotheses_summary"] == []


class TestBuildTraceViewWithHistory:
    def _make_trace_with_rounds(self, decisions):
        scen = MockScenario()
        trace = Trace(scen=scen)

        for i, decision in enumerate(decisions):
            hyp = Hypothesis(
                hypothesis=f"Hypothesis {i}",
                reason=f"Reason {i}",
                concise_reason=f"Concise reason {i}",
                concise_observation=f"Obs {i}",
                concise_justification=f"Just {i}",
                concise_knowledge=f"Knowledge {i}",
            )
            exp = MockExperiment(hyp, result={"IC": 0.05 - i * 0.01})
            fb = HypothesisFeedback(
                reason=f"Round {i} feedback",
                decision=decision,
                code_change_summary=f"Change {i}",
            )
            trace.sync_dag_parent_and_hist((exp, fb), i)

        return trace

    def test_sota_extraction(self):
        trace = self._make_trace_with_rounds([True, False, False])
        view = build_trace_view(trace)

        assert view["total_rounds"] == 3
        assert view["sota"] is not None
        assert view["sota"]["round_id"] == 0
        assert view["sota"]["hypothesis"] == "Hypothesis 0"

    def test_recent_rounds(self):
        trace = self._make_trace_with_rounds([True, False, True, False, False])
        view = build_trace_view(trace, recent_rounds=3)

        assert len(view["recent_rounds"]) == 3
        assert view["recent_rounds"][0]["round_id"] == 2
        assert view["recent_rounds"][-1]["round_id"] == 4

    def test_failed_hypotheses(self):
        trace = self._make_trace_with_rounds([False, False, True, False])
        view = build_trace_view(trace)

        failed = view["failed_hypotheses_summary"]
        assert len(failed) == 3
        assert "Concise reason 0" in failed

    def test_no_sota_when_all_fail(self):
        trace = self._make_trace_with_rounds([False, False])
        view = build_trace_view(trace)

        assert view["sota"] is None

    def test_single_round(self):
        trace = self._make_trace_with_rounds([True])
        view = build_trace_view(trace)

        assert view["total_rounds"] == 1
        assert view["sota"]["round_id"] == 0
        assert len(view["recent_rounds"]) == 1
        assert view["failed_hypotheses_summary"] == []
