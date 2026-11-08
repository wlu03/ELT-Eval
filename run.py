"""Run the ELT-Bench Codex agent on an unmodified external benchmark."""

import argparse
import importlib
import traceback
from datetime import datetime, timezone
from uuid import uuid4

from runner import ROOT, SOURCE, command, write_json

ADAPTERS = {
    "spider2-dbt": "adapter.spider2_dbt",
    "dacomp-de": "adapter.dacomp_de",
    "data-eng-bench": "adapter.data_eng_bench",
    "ade-bench": "adapter.ade_bench",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=ADAPTERS, required=True)
    parser.add_argument("--task", required=True, help="Original task ID, or taskall.")
    parser.add_argument("--model", help="Defaults to the ELT-Bench runner's model.")
    parser.add_argument("--db", choices=["duckdb", "snowflake"], default="duckdb")
    args = parser.parse_args(argv)
    try:
        adapter = importlib.import_module(ADAPTERS[args.benchmark])
        available = adapter.tasks(args.db)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.task == "taskall":
        selected = available
    elif args.task in available:
        selected = [args.task]
    else:
        parser.error(f"Unknown task or unsupported backend: {args.task} ({args.db})")
    if not selected:
        parser.error("No supported tasks found. Check the benchmark source/data setup.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = ROOT / "runs" / args.benchmark / run_id
    output.mkdir(parents=True)
    revisions = {}
    for repo in (SOURCE / "ELT-Bench", adapter.REPO):
        revisions[repo.name] = command("git", "-C", repo, "rev-parse", "HEAD").stdout.strip()
    write_json(output / "manifest.json", {
        "benchmark": args.benchmark, "tasks": selected, "db": args.db,
        "requested_model": args.model, "source_revisions": revisions,
        "agent_implementation": "ELT-Bench++/agents/codex_agent/run.py",
    })

    summary = []
    for task_id in selected:
        attempt = output / task_id
        attempt.mkdir()
        print(f"Running {args.benchmark}/{task_id}", flush=True)
        try:
            result = adapter.run(task_id, attempt, args.model, args.db)
        except Exception:
            result = {"status": "error", "error": traceback.format_exc()}
        result.update(benchmark=args.benchmark, task_id=task_id, db=args.db)
        write_json(attempt / "result.json", result)
        summary.append(result)
        write_json(output / "summary.json", summary)
        print(f"{task_id}: {result['status']}", flush=True)
        if result["status"] == "error":
            print(result.get("error", "Native harness error; see result.json."), flush=True)
    print(f"Results: {output}")
    # Native task failures are valid evaluations; infrastructure failures are not.
    return int(any(result["status"] == "error" for result in summary))


if __name__ == "__main__":
    raise SystemExit(main())
