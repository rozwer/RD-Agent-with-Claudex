"""2-round verification scenario for stub adapters (Task 1A.3.1)."""
import uuid

import pytest
from pathlib import Path

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


@pytest.fixture
def run_env(tmp_path):
    base = tmp_path / "artifacts"
    run_dir = create_run_dir(base, "test_run")
    create_round_dir(run_dir, 0)
    create_round_dir(run_dir, 1)
    scen = MockScenario()
    return run_dir, scen


def run_one_round(run_dir, scen, trace, round_idx, hyp_gen, h2e, coder, summarizer):
    hypothesis = hyp_gen.gen(trace)
    exp = h2e.convert(hypothesis, trace)
    exp = coder.develop(exp)

    exp.result = {"IC": 0.12 if round_idx == 0 else 0.08, "ICIR": 0.45 if round_idx == 0 else 0.32}
    exp.stdout = f"round{round_idx + 1} ok"

    save_artifact(
        run_dir / f"round_{round_idx}" / "run_result.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "run_result",
            "artifact_id": str(uuid.uuid4()),
            "experiment_ref": f"round_{round_idx}/experiment.json",
            "status": "success",
            "result": exp.result,
            "stdout": exp.stdout,
        },
    )

    feedback = summarizer.generate_feedback(exp, trace)
    return exp, feedback


class TestTwoRoundScenario:
    def test_full_2_round(self, run_env):
        run_dir, scen = run_env
        trace = Trace(scen=scen)

        hyp_gen = ClaudeCodeFactorHypothesisGenAdapter(scen, run_dir=run_dir)
        h2e = ClaudeCodeFactorH2EAdapter(run_dir=run_dir)
        coder = ClaudeCodeFactorCoderAdapter(scen, run_dir=run_dir)
        summarizer = ClaudeCodeFactorSummarizerAdapter(scen, run_dir=run_dir)

        # ====== ROUND 1 ======
        exp1, fb1 = run_one_round(run_dir, scen, trace, 0, hyp_gen, h2e, coder, summarizer)
        trace.sync_dag_parent_and_hist((exp1, fb1), 0)

        # Hypothesis has 6 required fields
        h = exp1.hypothesis
        for field in [
            "hypothesis",
            "reason",
            "concise_reason",
            "concise_observation",
            "concise_justification",
            "concise_knowledge",
        ]:
            assert getattr(h, field, None) not in (None, "")

        assert len(exp1.sub_tasks) == 1
        assert exp1.sub_workspace_list[0].file_dict.get("factor.py") is not None
        assert fb1.decision is True
        assert len(exp1.based_experiments) >= 1

        # CoSTEER feedback
        assert exp1.prop_dev_feedback is not None
        assert len(exp1.prop_dev_feedback) == 1
        assert exp1.prop_dev_feedback[0].final_decision is True

        # ====== ROUND 2 ======
        exp2, fb2 = run_one_round(run_dir, scen, trace, 1, hyp_gen, h2e, coder, summarizer)
        trace.sync_dag_parent_and_hist((exp2, fb2), 1)

        assert fb2.decision is False
        assert len(trace.hist) == 2
        assert len(trace.dag_parent) == 2
        assert len(exp2.based_experiments) >= 1

        # Verify artifact files
        for round_idx in [0, 1]:
            round_dir = run_dir / f"round_{round_idx}"
            assert (round_dir / "hypothesis.json").exists()
            assert (round_dir / "experiment.json").exists()
            assert (round_dir / "feedback.json").exists()
            assert (round_dir / "run_result.json").exists()

        # SOTA is Round 1
        sota_hyp, sota_exp = trace.get_sota_hypothesis_and_experiment()
        assert sota_exp is exp1
