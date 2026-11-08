"""Container transport and shared ELT-Bench agent session; no benchmark grading."""

import ast
import asyncio
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RUNTIME_IMAGE = "elt-eval-duckdb"
_AGENT_LOCK = threading.Lock()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def command(*args, cwd=None, log=None, check=True, timeout=None, input=None):
    """Run host-controlled commands; agent commands only use docker exec."""
    try:
        result = subprocess.run(
            [str(arg) for arg in args], cwd=cwd, input=input, text=True,
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if log:
            Path(log).write_text(f"Command timed out after {timeout}s.\n")
        raise RuntimeError(f"Command timed out after {timeout}s; see {log}") from exc
    if log:
        Path(log).write_text(result.stdout + result.stderr)
    if check and result.returncode:
        # Do not echo argv: container env arguments can contain credentials.
        raise RuntimeError(
            f"{args[0]} exited with {result.returncode}; "
            f"{('see ' + str(log)) if log else result.stderr.strip()[-2000:]}"
        )
    return result


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark asset: {path}. Run bash setup.sh and the "
            "upstream dataset setup documented in README.md."
        )
    return path


def within(root, relative):
    """Resolve a benchmark-relative path without allowing escapes."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path must stay within {root}: {relative}")
    return path


def native_functions(path, names, namespace=None):
    """Load pure upstream helpers without importing an unrelated agent runtime."""
    path = require(path)
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise RuntimeError(f"Upstream helper interface changed: {path}")
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + functions,
        type_ignores=[],
    )
    scope = dict(namespace or {})
    exec(compile(module, str(path), "exec"), scope)
    return scope


@cache
def build_runtime():
    print("Preparing the DuckDB runtime image...", flush=True)
    command("docker", "build", "-t", RUNTIME_IMAGE, "-",
            input=(ROOT / "Dockerfile").read_text(), timeout=1800)


@contextmanager
def container(image, workdir, mounts=(), env=None, limits=()):
    name = f"elt-eval-{uuid4().hex[:12]}"
    args = ["docker", "run", "-d", "--name", name, "--entrypoint", "sleep",
            "--workdir", workdir, *limits]
    for source, target, readonly in mounts:
        mount = f"type=bind,source={require(source).resolve()},target={target}"
        if readonly:
            mount += ",readonly"
        args.extend(["--mount", mount])
    for key, value in (env or {}).items():
        args.extend(["--env", f"{key}={value}"])
    try:
        command(*args, image, "infinity", timeout=300)
        yield name
    finally:
        command("docker", "rm", "-f", name, check=False, timeout=60)


@cache
def elt_runner():
    path = require(SOURCE / "ELT-Bench/agents/codex_agent/run.py")
    spec = importlib.util.spec_from_file_location("elt_eval_codex", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep per-attempt traces in runs/, not a second transcript in source/.
    for handler in list(module.logger.handlers):
        if isinstance(handler, logging.FileHandler):
            module.logger.removeHandler(handler)
            handler.close()
    return module


def agent_defaults():
    with patch.object(sys, "argv", ["elt-agent"]):
        return elt_runner().parse_args()


def run_agent(container_name, prompt, workdir, output, model=None,
              timeout=4000, protected=(), writable_roots=None, extra_instructions=""):
    """Use ELT's model loop/tools inside an already initialized container."""
    module = elt_runner()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    defaults = agent_defaults()
    model = model or defaults.model
    roots = tuple(PurePosixPath(p) for p in (writable_roots or (workdir,)))
    protected = tuple(PurePosixPath(p) for p in protected)
    (output / "prompt.txt").write_text(prompt)

    def validate_path(path):
        if not PurePosixPath(path).is_absolute():
            return "File-edit paths must be absolute."
        resolved = command("docker", "exec", container_name, "realpath", "-m", "--", path,
                           check=False, timeout=30)
        if resolved.returncode:
            return f"Cannot resolve path: {path}"
        actual = PurePosixPath(resolved.stdout.strip())
        if not any(actual.is_relative_to(root) for root in roots):
            return f"Edits must remain under: {', '.join(map(str, roots))}"
        if any(actual == item or actual.is_relative_to(item) for item in protected):
            return f"Task-protected path: {path}"
        return None

    async def execute():
        recorder = module.TrajectoryRecorder()
        result = None
        system = (
            "You are a data engineer skilled in databases, SQL, and building ELT pipelines.\n"
            f"Your working directory is {workdir}. Your shell and apply_patch tools "
            "execute inside this task's container.\n"
            f"Permitted file-edit roots: {', '.join(map(str, roots))}.\n"
            "Follow the original task instructions and restrictions. Use shell for "
            "inspection and execution and apply_patch for file edits.\n"
            "Use absolute paths with apply_patch. Read applicable AGENTS.md and "
            "other project instruction files before editing.\n"
            "First, briefly state what you are going to do next. Then issue the tool "
            "call and wait for its result before continuing."
        )
        if protected:
            system += "\nDo not modify: " + ", ".join(map(str, protected))
        if extra_instructions:
            system += "\n\n" + extra_instructions
        (output / "system_prompt.txt").write_text(system)
        try:
            async with asyncio.timeout(timeout):
                agent = module.Agent(
                    name="elt-bench-agent", model=model,
                    model_settings=module.ModelSettings(),
                    instructions=system, tools=module.build_tools(container_name),
                )
                result = module.Runner.run_streamed(
                    agent, input=prompt, max_turns=defaults.max_turns,
                )
                async for event in result.stream_events():
                    if event.type == "run_item_stream_event":
                        recorder.record_run_item(event.item)
        except Exception as exc:
            recorder.record_error(exc)
        finally:
            if result is not None:
                with suppress(Exception):
                    recorder.finalize_usage(result)
                if not recorder.final_text:
                    recorder.final_text = str(getattr(result, "final_output", "") or "")
        record = {
            "model": model, "finished": not recorder.is_error,
            "num_turns": recorder.num_turns, "usage": recorder.usage,
            "result": recorder.final_text, "trajectory": recorder.turns,
        }
        write_json(output / "agent.json", record)
        return record

    # The source module's tools resolve these globals at call time. Serialize
    # sessions and restore them afterward; never invoke its ELT setup/main().
    with _AGENT_LOCK, patch.object(module, "CONTAINER_WORKDIR", workdir), \
            patch.object(module, "_enforce_workspace", validate_path):
        return asyncio.run(execute())


def run_project(project, task_id, prompt, output, model, protected=(), extra_instructions=""):
    build_runtime()
    workspace = output / "submission" / task_id
    shutil.copytree(require(project), workspace)  # Never merge with an old attempt.
    mounts = [(workspace, "/workspace", False)]
    for relative in protected:
        mounts.append((require(within(project, relative)), f"/workspace/{relative}", True))
    with container(RUNTIME_IMAGE, "/workspace", mounts,
                   {"DBT_PROFILES_DIR": "/workspace"}) as name:
        result = run_agent(name, prompt, "/workspace", output, model,
                           protected=[f"/workspace/{p}" for p in protected],
                           extra_instructions=extra_instructions)
    return workspace, result  # All agent processes have stopped before grading.
