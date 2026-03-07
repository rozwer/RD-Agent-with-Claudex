# RD-Agent-with-Claudex

Microsoft RD-Agent fork: OpenAI API replaced with Claude Code + subagents + Codex.

## Architecture

**Control Inversion**: Claude Code orchestrates Python/Qlib as tools (not Python calling LLM API).

RDLoop 5-slot adapter pattern bridges Claude Code components to RD-Agent data structures:

| Slot | Adapter | Component |
|------|---------|-----------|
| hypothesis_gen | ClaudeCodeFactorHypothesisGenAdapter | Planner subagent |
| hypothesis2experiment | ClaudeCodeFactorH2EAdapter | Planner subagent |
| coder | ClaudeCodeFactorCoderAdapter | Codex |
| runner | QlibFactorRunner (existing) | Bash / Qlib |
| summarizer | ClaudeCodeFactorSummarizerAdapter | Evaluator subagent |

## Key Paths

- `rdagent/adapters/` — Adapter layer (Phase 1A-1B)
- `rdagent/adapters/factor/` — Factor-specific adapters (planner, evaluator, trace_view, coder, h2e)
- `rdagent/adapters/artifact_utils.py` — Artifact directory management
- `rdagent/oai/backend/` — LLM backends (LiteLLM, Claude Code shim)
- `rdagent/oai/llm_conf.py` — LLM configuration (defaults to Claude)
- `rdagent/core/` — Data structures (Hypothesis, Trace, Experiment) - unchanged
- `rdagent/components/workflow/rd_loop.py` — Main RDLoop
- `rdagent/scenarios/qlib/` — Qlib scenario implementation

## Artifact Structure

```
.claude/artifacts/rdloop/<run_id>/
  trace.json              # Experiment history (SSOT)
  round_manifest.json     # Run-level manifest
  round_<N>/
    manifest.json         # Round status + timestamps
    hypothesis.json       # Planner output
    experiment.json       # Experiment specification
    run_result.json       # Backtest results
    feedback.json         # Evaluator output
    implementations/
      factor.py           # Generated factor code
```

## Configuration

Default backend: `LiteLLMAPIBackend` with `anthropic/claude-sonnet-4-20250514`.
Set `ANTHROPIC_API_KEY` environment variable for API access.

## Testing

```bash
cd RD-Agent-with-Claudex
source .venv/bin/activate
pytest test/adapters/ -v  # Adapter unit tests (38 tests)
```

## Conventions

- Documentation in concise Japanese
- MIT license (inherited from upstream)
- Brand: `with-Claudex`
- Phase 1 scope: factor scenario only
