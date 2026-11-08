"""Spider's supplied dbt project -> native results_metadata.jsonl evaluation."""

import json
import re
import runpy
import shutil

import yaml

from runner import (
    SOURCE, RUNTIME_IMAGE, agent_defaults, command, container, require, run_project, within,
)

REPO = SOURCE / "Spider2"
BENCH = REPO / "spider2-dbt"

TOOL_INSTRUCTIONS = (
    "The Spider prompt preserves its native action descriptions and response format. "
    "Adapt only that tool protocol: invoke shell for Bash; invoke Python's duckdb "
    "library through shell for LOCAL_DB_SQL; invoke apply_patch with absolute paths "
    "for CreateFile/EditFile; return the required DuckDB filename in your final "
    "response for Terminate. Use actual tool calls instead of textual Thought/Action "
    "messages. All native task requirements and restrictions remain in force."
)


def records(path):
    return [json.loads(line) for line in require(path).read_text().splitlines() if line.strip()]


def tasks(db):
    if db != "duckdb":
        return []
    return sorted(row["instance_id"] for row in records(BENCH / "examples/spider2-dbt.jsonl"))


def native_prompt(task):
    """Render Spider's complete DBT prompt, including all execution/edit rules."""
    directory = REPO / "methods/spider-agent-dbt/spider_agent/agent"
    prompts = runpy.run_path(str(require(directory / "prompts.py")))
    actions = runpy.run_path(str(require(directory / "action.py")))
    names = ("Bash", "Terminate", "CreateFile", "EditFile", "LOCAL_DB_SQL")
    return prompts["DBT_SYSTEM"].format(
        work_dir="/workspace", max_steps=agent_defaults().max_turns,
        action_space="".join(actions[name].get_action_description() for name in names),
        task=task["instruction"],
    )


def parse_score(stdout):
    # This is the native evaluator's final aggregate, not the agent's claimed status.
    lines = stdout.strip().splitlines()
    match = re.search(r"^([0-9.]+)\s+(\d+)\s+(\d+)\s*$", lines[-1]) if lines else None
    if not match or int(match[3]) != 1 or int(match[2]) not in (0, 1):
        raise RuntimeError("Spider did not report exactly one evaluated task; see evaluation.log")
    score = float(match[1])
    if score != int(match[2]):
        raise RuntimeError("Inconsistent native Spider aggregate")
    return {"score": score, "correct": int(match[2]), "total": int(match[3])}


def run(task_id, output, model, db):
    if task_id not in tasks(db):
        raise ValueError(f"Unsupported Spider task/backend: {task_id}/{db}")
    project = BENCH / "examples" / task_id
    suite = BENCH / "evaluation_suite"
    task = next(row for row in records(BENCH / "examples/spider2-dbt.jsonl")
                if row["instance_id"] == task_id)
    config = yaml.safe_load(require(project / "dbt_project.yml").read_text())
    profiles = yaml.safe_load(require(project / "profiles.yml").read_text())
    profile = profiles[config["profile"]]
    target = profile["outputs"][profile["target"]]
    if target["type"] != "duckdb":
        raise ValueError("Only the native DuckDB variant is supported for Spider 2.0-DBT")
    database = target["path"]
    require(within(project, database))

    gold_record = next(row for row in records(suite / "gold/spider2_eval.jsonl")
                       if row["instance_id"] == task_id)
    comparisons = gold_record["evaluation"]
    if isinstance(comparisons, dict):
        comparisons = [comparisons]
    for comparison in comparisons:
        if comparison["func"] != "duckdb_match":
            raise ValueError(f"Unsupported Spider submission type: {comparison['func']}")
        files = comparison["parameters"]["gold"]
        for filename in ([files] if isinstance(files, str) else files):
            require(within(suite / "gold" / task_id, filename))

    _, agent = run_project(project, task_id, native_prompt(task), output, model,
                           extra_instructions=TOOL_INSTRUCTIONS)
    submission = output / "submission"
    (submission / "results_metadata.jsonl").write_text(json.dumps({
        "instance_id": task_id, "answer_type": "file", "answer_or_path": database,
    }) + "\n")

    # Gold is never mounted in the agent container. Use a separate writable copy
    # because the original evaluator opens DuckDB files in read/write mode.
    gold = output / "gold"
    gold.mkdir()
    shutil.copy2(suite / "gold/spider2_eval.jsonl", gold)
    shutil.copytree(suite / "gold" / task_id, gold / task_id)
    with container(RUNTIME_IMAGE, "/evaluation", [
        (suite, "/evaluation", True), (submission, "/submission", False),
        (gold, "/gold", False),
    ]) as name:
        evaluated = command(
            "docker", "exec", name, "python", "evaluate.py",
            "--result_dir", "/submission", "--gold_dir", "/gold",
            log=output / "evaluation.log", timeout=3600,
        )
    return {"status": "evaluated", "agent_finished": agent["finished"],
            "native_result": parse_score(evaluated.stdout)}
