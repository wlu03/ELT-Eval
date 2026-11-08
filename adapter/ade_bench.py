"""An ELT agent hook for ADE's original setup and evaluation harness."""

import sys
import time
from contextlib import chdir

import yaml

from runner import SOURCE, agent_defaults, command, require, run_agent

REPO = SOURCE / "ade-bench"


def tasks(db):
    selected = []
    for path in require(REPO / "tasks").glob("*/task.yaml"):
        task = yaml.safe_load(path.read_text())
        supported = any(v["db_type"] == db and v["project_type"] == "dbt"
                        for v in task.get("variants", []))
        base = any(p["key"] == "base" for p in task.get("prompts", []))
        if task.get("status") == "ready" and supported and base:
            selected.append(path.parent.name)
    return sorted(selected)


def run(task_id, output, model, db):
    if task_id not in tasks(db):
        raise ValueError(f"Unsupported ADE task/backend: {task_id}/{db}")
    sys.path.insert(0, str(REPO))
    from ade_bench.agents.agent_factory import AgentFactory
    from ade_bench.agents.agent_name import AgentName
    from ade_bench.agents.base_agent import AgentResult, BaseAgent
    from ade_bench.harness import Harness
    from ade_bench.harness_models import FailureMode
    from ade_bench.utils.timeout_manager import TimeoutManager

    task = yaml.safe_load((REPO / "tasks" / task_id / "task.yaml").read_text())
    defaults = agent_defaults()
    model = model or defaults.model
    timeouts = TimeoutManager.get_default_timeouts()
    agent_timeout = task.get("max_agent_timeout_sec") or timeouts.agent_execution

    class ELTAgent(BaseAgent):
        NAME = AgentName.OPENAI_CODEX

        def perform_task(self, task_prompt, session, logging_dir=None, task_name=None):
            started = time.monotonic()
            # Native logging_dir is unique even if one task has multiple variants.
            logs = logging_dir or output
            # ADE installs this for Codex CLI, which normally reads it itself.
            # Our SDK agent needs the same instructions supplied explicitly.
            config = command(
                "docker", "exec", session.container.name, "bash", "-c",
                "if [ -f /app/AGENTS.md ]; then cat /app/AGENTS.md; fi", timeout=30,
            ).stdout
            result = run_agent(session.container.name, task_prompt, "/app", logs,
                               model, timeout=agent_timeout,
                               extra_instructions=("Project instructions (/app/AGENTS.md):\n" + config
                                                   if config else ""))
            command("docker", "cp", f"{session.container.name}:/app", logs / "submission",
                    timeout=max(1, timeouts.cleanup - 1))
            usage = result["usage"] or {}
            return AgentResult(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                num_turns=result["num_turns"], model_name=result["model"],
                runtime_ms=int((time.monotonic() - started) * 1000),
                failure_mode=(FailureMode.NONE if result["finished"]
                              else FailureMode.UNKNOWN_AGENT_ERROR),
            )

    class ELTFactory(AgentFactory):
        @property
        def agent_name(self):
            return AgentName.OPENAI_CODEX

        def get_agent(self, **kwargs):
            return ELTAgent(**kwargs)

    with chdir(REPO):
        native = Harness(
            dataset_path=REPO / "tasks", output_path=output, run_id="native",
            agent_factory=ELTFactory(), model_name=model,
            task_ids=[f"{task_id}.base"], db_type=db, project_type="dbt",
            plugin_set_names=["none"], n_concurrent_trials=1, n_attempts=1,
            max_episodes=defaults.max_turns, upload_results=False,
        ).run()
    if not native.results:
        raise RuntimeError("ADE did not evaluate the selected task")
    errors = [r.failure_mode.value for r in native.results if r.failure_mode.is_error()]
    return {"status": "error" if errors else "evaluated",
            "agent_implementation": "ELT-Bench++", "harness_errors": errors,
            "native_result": native.model_dump(mode="json")}
