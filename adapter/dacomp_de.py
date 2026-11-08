"""Original DE-Impl/DE-Evol bundles and prompts, graded by native cfs."""

import json
import logging
import os
import shutil

import yaml

from runner import (
    SOURCE, RUNTIME_IMAGE, command, container, native_functions,
    require, run_project, within,
)

REPO = SOURCE / "DAComp"
BENCH = REPO / "dacomp-de"


def tasks(db):
    if db != "duckdb":
        return []
    directory = require(BENCH / "tasks")
    return sorted(path.name for path in directory.iterdir() if path.is_dir()
                  and path.name.startswith(("dacomp-de-impl-", "dacomp-de-evol-")))


def native_prompt(project, task_id):
    scope = native_functions(
        REPO / "methods/de-agent/evaluation/benchmarks/dacomp/run_infer_de.py",
        {"get_task_files", "create_de_task_prompt"},
        {"os": os, "logger": logging.getLogger("dacomp-adapter")},
    )
    return scope["create_de_task_prompt"]({
        "source_dir": str(project),
        "task_type": "evol" if "-evol-" in task_id else "impl",
    }, "en")


def config_id(task_id, examples):
    resolve = native_functions(BENCH / "evaluation_suite/utils.py", {"resolve_example_id"})
    result, error = resolve["resolve_example_id"](task_id, examples)
    if error:
        raise ValueError(error)
    return result


def run(task_id, output, model, db):
    if task_id not in tasks(db):
        raise ValueError(f"Unsupported DAComp task/backend: {task_id}/{db}")
    project = require(BENCH / "tasks" / task_id)
    suite = BENCH / "evaluation_suite"
    for filename in ("run.py", "config/layer_dependencies.yaml",
                     "question.md" if "-evol-" in task_id else "docs/data_contract.yaml"):
        require(project / filename)
    if not list(project.glob("*_start.duckdb")):
        raise FileNotFoundError(f"No starting DuckDB database in {project}")
    config = yaml.safe_load(require(suite / "evaluation_config_compare.yaml").read_text())
    identifier = config_id(task_id, config["examples"])
    filename = config["examples"][identifier].get("database_file", "database.duckdb")
    require(within(suite / "gold" / identifier, filename))

    protected = ("run.py",) if "-evol-" in task_id else ()
    _, agent = run_project(project, task_id, native_prompt(project, task_id),
                           output, model, protected=protected)
    gold = output / "gold"
    gold.mkdir()
    shutil.copytree(suite / "gold" / identifier, gold / identifier)
    evaluation = output / "evaluation"
    evaluation.mkdir()
    with container(RUNTIME_IMAGE, "/evaluation", [
        (suite, "/evaluation", True), (output / "submission", "/submission", False),
        (gold, "/gold", False), (evaluation, "/results", False),
    ]) as name:
        command(
            "docker", "exec", name, "python", "evaluate.py", "single",
            "--pred_dir", "/submission", "--example_id", task_id,
            "--gold_dir", "/gold", "--config", "evaluation_config_compare.yaml",
            "--mode", "cfs", "--force-rebuild", "--output", "/results/native.json",
            log=output / "evaluation.log", timeout=3600,
        )
    native = json.loads(require(evaluation / "native.json").read_text())
    if native.get("error"):
        raise RuntimeError(f"Native DAComp evaluator error: {native['error']}")
    if "final_score" not in native:
        raise RuntimeError("Native DAComp evaluator did not return final_score")
    return {"status": "evaluated", "agent_finished": agent["finished"],
            "native_result": native}
