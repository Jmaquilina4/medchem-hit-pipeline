"""Pipeline engine: the stage contract, content-addressed cache, and DAG runner.

Invariant: nothing in this package (or anywhere under ``src/``) makes LLM calls.
Agentic orchestration lives in ``skills/`` on top of the CLI.
"""
