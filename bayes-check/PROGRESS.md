# bayes-check — PROGRESS

Append-only log so state survives compaction/resume.

- M0: Read GOAL.md in full; read plugin contract code (plugins.py, plugins_content.py,
  plugins_dispatch.py, plugin_dev.py, model_tools.py, delegate_tool*.py, session_context.py).
  Built test env (`python -m pm.build_env --out .venv --group dev --group test`).
- M1: DESIGN.md written before any code (6 code/spec discrepancies, 13 new-judgment items).
- M2: Implementation (scorer, schemas, local_mode, extract, footer, ledger, state, config, tools,
  hooks, commands, __init__, plugin.yaml, SKILL.md). Doctor --ci exit 0, zero warnings.
  Found D7 (handler gets no turn_id) → call_id binding via post_tool_call.
- M3: 102 tests (unit + real-runtime integration via Hermes discovery in a temp HERMES_HOME),
  all passing; mutation check confirms the verify tests can fail. ruff clean.
- M4: README with verify-in-dev results (both flagged items CONFIRMED, with corrections D5/D7).
