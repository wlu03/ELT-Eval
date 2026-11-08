"""Native task images, healthchecks, instructions, and test.sh verifier."""

import os
import time
import tomllib
from uuid import uuid4

from runner import SOURCE, command, container, require, run_agent

REPO = SOURCE / "data-eng-bench"


def tasks(db):
    return sorted(path.parent.name for path in require(REPO / "tasks").glob("*/task.toml"))


def initialize(name, health):
    if health:
        attempts = max(1, health.get("retries", 1))
        for attempt in range(attempts):
            result = command("docker", "exec", name, "bash", "-c", health["command"],
                             timeout=health.get("timeout_sec", 600), check=False)
            if result.returncode == 0:
                break
            if attempt + 1 == attempts:
                raise RuntimeError(f"Native healthcheck failed: {result.stderr[-2000:]}")
            time.sleep(health.get("interval_sec", 10))
    # ELT's shell tool opens login shells, which wait for this native marker.
    command("docker", "exec", name, "touch", "/tmp/entrypoint_ready")


def run(task_id, output, model, db):
    if task_id not in tasks(db) or db not in ("duckdb", "snowflake"):
        raise ValueError(f"Unsupported data-eng-bench task/backend: {task_id}/{db}")
    task = require(REPO / "tasks" / task_id)
    config = tomllib.loads((task / "task.toml").read_text())
    environment = config["environment"]
    image = f"elt-eval-data-eng-{task_id}"
    command("docker", "build", "-t", image, task / "environment",
            timeout=environment.get("build_timeout_sec", 3000), log=output / "build.log")
    env = {"DB_TYPE": db}
    if db == "snowflake":
        env.update({key: value for key, value in os.environ.items() if key.startswith("SNOWFLAKE_")})
    limits = ["--cpus", str(environment.get("cpus", 2)),
              "--memory", f"{environment.get('memory_mb', 8000)}m"]
    snapshot = f"elt-eval-submission:{uuid4().hex}"
    cleanup_done = False
    try:
        with container(image, "/app", env=env, limits=limits) as name:
            try:
                initialize(name, environment.get("healthcheck"))
                agent = run_agent(name, (task / "instruction.md").read_text(), "/app",
                                   output, model, timeout=config["agent"].get("timeout_sec", 4000))
                command("docker", "cp", f"{name}:/app", output / "submission", timeout=300)
                # Transfer filesystem state to a fresh verifier container. No
                # agent/background processes survive to observe hidden tests.
                command("docker", "commit", name, snapshot, timeout=300)
            except BaseException:
                if db == "snowflake":
                    command("docker", "exec", name, "python3", "/cleanup_snowflake.py",
                            check=False, timeout=120)
                    cleanup_done = True
                raise
        with container(snapshot, "/app", limits=limits) as name:
            try:
                command("docker", "cp", task / "tests", f"{name}:/tests")
                evaluated = command(
                    "docker", "exec", name, "bash", "/tests/test.sh",
                    timeout=config["verifier"].get("timeout_sec", 3000),
                    check=False, log=output / "evaluation.log",
                )
                evaluation = output / "evaluation"
                evaluation.mkdir()
                command("docker", "cp", f"{name}:/logs/verifier/.", evaluation)
                reward = require(evaluation / "reward.txt").read_text().strip()
                if reward not in {"0", "1"}:
                    raise RuntimeError(f"Unexpected native reward: {reward}")
                if reward == "1" and evaluated.returncode:
                    raise RuntimeError("Verifier exited with an error despite reward=1")
                return {"status": "evaluated", "agent_finished": agent["finished"],
                        "native_reward": int(reward), "verifier_exit_code": evaluated.returncode}
            finally:
                if db == "snowflake":
                    command("docker", "exec", name, "python3", "/cleanup_snowflake.py",
                            check=False, timeout=120)
                    cleanup_done = True
    finally:
        # A verifier launch failure must not leave the native Snowflake clone.
        if db == "snowflake" and not cleanup_done:
            command("docker", "run", "--rm", "--entrypoint", "python3", snapshot,
                    "/cleanup_snowflake.py", check=False, timeout=180)
        command("docker", "image", "rm", snapshot, check=False, timeout=120)
