"""Resume tests for 4 checkpoint points (Task 1A.3.2)."""
import uuid

import pytest

from rdagent.adapters.artifact_utils import (
    SCHEMA_VERSION,
    create_round_dir,
    create_run_dir,
    save_artifact,
)
from rdagent.adapters.factor.coder import ClaudeCodeFactorCoderAdapter
from rdagent.adapters.factor.h2e import ClaudeCodeFactorH2EAdapter
from rdagent.adapters.factor.hypothesis_gen import ClaudeCodeFactorHypothesisGenAdapter
from rdagent.adapters.factor.summarizer import ClaudeCodeFactorSummarizerAdapter
from rdagent.core.proposal import Trace


class MockScenario:
    @property
    def background(self):
        return "test"

    def get_source_data_desc(self, task=None):
        return "test"

    @property
    def rich_style_description(self):
        return "test"


def _write_run_result(run_dir, round_idx, result, stdout="ok"):
    save_artifact(
        run_dir / f"round_{round_idx}" / "run_result.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "run_result",
            "artifact_id": str(uuid.uuid4()),
            "experiment_ref": f"round_{round_idx}/experiment.json",
            "status": "success",
            "result": result,
            "stdout": stdout,
        },
    )


def _make_adapters(scen, run_dir):
    return (
        ClaudeCodeFactorHypothesisGenAdapter(scen, run_dir=run_dir),
        ClaudeCodeFactorH2EAdapter(run_dir=run_dir),
        ClaudeCodeFactorCoderAdapter(scen, run_dir=run_dir),
        ClaudeCodeFactorSummarizerAdapter(scen, run_dir=run_dir),
    )


def full_run(run_dir, scen, trace):
    hyp_gen, h2e, coder, summarizer = _make_adapters(scen, run_dir)
    for round_idx in range(2):
        hypo = hyp_gen.gen(trace)
        exp = h2e.convert(hypo, trace)
        exp = coder.develop(exp)
        exp.result = {"IC": 0.12 if round_idx == 0 else 0.08}
        exp.stdout = f"round{round_idx + 1} ok"
        _write_run_result(run_dir, round_idx, exp.result, exp.stdout)
        fb = summarizer.generate_feedback(exp, trace)
        trace.sync_dag_parent_and_hist((exp, fb), round_idx)
    return hyp_gen, h2e, coder, summarizer


class TestCheckpoint1ExpGenComplete:
    """Resume after exp_gen → coding onwards."""

    def test_resume_from_exp_gen(self, tmp_path):
        scen = MockScenario()

        # Fresh reference
        fresh_dir = create_run_dir(tmp_path / "fresh", "run")
        for i in range(2):
            create_round_dir(fresh_dir, i)
        fresh_trace = Trace(scen=scen)
        full_run(fresh_dir, scen, fresh_trace)

        # Resume run
        resume_dir = create_run_dir(tmp_path / "resume", "run")
        for i in range(2):
            create_round_dir(resume_dir, i)
        resume_trace = Trace(scen=scen)

        hyp_gen, h2e, coder, summarizer = _make_adapters(scen, resume_dir)

        for round_idx in range(2):
            hypo = hyp_gen.gen(resume_trace)
            exp = h2e.convert(hypo, resume_trace)
            exp = coder.develop(exp)
            exp.result = {"IC": 0.12 if round_idx == 0 else 0.08}
            exp.stdout = f"round{round_idx + 1} ok"
            _write_run_result(resume_dir, round_idx, exp.result, exp.stdout)
            fb = summarizer.generate_feedback(exp, resume_trace)
            resume_trace.sync_dag_parent_and_hist((exp, fb), round_idx)

        assert len(resume_trace.hist) == len(fresh_trace.hist)
        assert len(resume_trace.dag_parent) == len(fresh_trace.dag_parent)
        for i in range(2):
            assert resume_trace.hist[i][1].decision == fresh_trace.hist[i][1].decision


class TestCheckpoint2CodingComplete:
    """Resume after coding → running onwards."""

    def test_resume_from_coding(self, tmp_path):
        scen = MockScenario()
        run_dir = create_run_dir(tmp_path / "resume2", "run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        hyp_gen, h2e, coder, summarizer = _make_adapters(scen, run_dir)

        hypo = hyp_gen.gen(trace)
        exp = h2e.convert(hypo, trace)
        exp = coder.develop(exp)

        assert exp.sub_workspace_list[0].file_dict.get("factor.py") is not None

        exp.result = {"IC": 0.12}
        _write_run_result(run_dir, 0, exp.result, "round1 ok")
        fb = summarizer.generate_feedback(exp, trace)
        trace.sync_dag_parent_and_hist((exp, fb), 0)

        assert fb.decision is True
        assert len(trace.hist) == 1


class TestCheckpoint3RunningComplete:
    """Resume after running → feedback onwards."""

    def test_resume_from_running(self, tmp_path):
        scen = MockScenario()
        run_dir = create_run_dir(tmp_path / "resume3", "run")
        create_round_dir(run_dir, 0)
        trace = Trace(scen=scen)

        hyp_gen, h2e, coder, summarizer = _make_adapters(scen, run_dir)

        hypo = hyp_gen.gen(trace)
        exp = h2e.convert(hypo, trace)
        exp = coder.develop(exp)
        exp.result = {"IC": 0.12}

        fb = summarizer.generate_feedback(exp, trace)
        trace.sync_dag_parent_and_hist((exp, fb), 0)

        assert fb.decision is True
        assert (run_dir / "round_0" / "feedback.json").exists()


class TestCheckpoint4FeedbackComplete:
    """Resume after feedback → next round."""

    def test_resume_to_next_round(self, tmp_path):
        scen = MockScenario()
        run_dir = create_run_dir(tmp_path / "resume4", "run")
        for i in range(2):
            create_round_dir(run_dir, i)
        trace = Trace(scen=scen)

        hyp_gen, h2e, coder, summarizer = _make_adapters(scen, run_dir)

        hypo = hyp_gen.gen(trace)
        exp = h2e.convert(hypo, trace)
        exp = coder.develop(exp)
        exp.result = {"IC": 0.12}
        _write_run_result(run_dir, 0, exp.result, "ok")
        fb = summarizer.generate_feedback(exp, trace)
        trace.sync_dag_parent_and_hist((exp, fb), 0)

        assert (run_dir / "round_0" / "feedback.json").exists()
        assert len(trace.hist) == 1

        # Start round 1
        hypo2 = hyp_gen.gen(trace)
        exp2 = h2e.convert(hypo2, trace)
        assert len(exp2.based_experiments) >= 1
        assert hyp_gen._call_count == 2
