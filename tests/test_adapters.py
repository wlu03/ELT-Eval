"""Contract tests; set ELT_EVAL_DOCKER_TESTS=1 for real container checks."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import run
import runner
from adapter import ade_bench, dacomp_de, data_eng_bench, spider2_dbt


@contextmanager
def fake_container(*args, **kwargs):
    yield "test-container"


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "attempt"
        self.out.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def project(self, name="task"):
        project = self.root / name
        project.mkdir()
        (project / "run.py").write_text("print('original pipeline')\n")
        (project / "sample_start.duckdb").write_bytes(b"starting data")
        (project / "sql").mkdir()
        (project / "sql/model.sql").write_text("select 1")
        return project

    def test_fresh_attempt_and_readonly_evolution_runner(self):
        project = self.project()
        seen = []

        @contextmanager
        def capture(image, cwd, mounts, env):
            seen.append(mounts)
            yield "test-container"

        def agent(*args, **kwargs):
            workspace = seen[-1][0][0]
            self.assertEqual((workspace / "sql/model.sql").read_text(), "select 1")
            self.assertEqual(kwargs["extra_instructions"], "Native tool mapping")
            (workspace / "sql/model.sql").write_text("select 2")
            return {"finished": True}

        with patch.object(runner, "build_runtime"), patch.object(runner, "container", capture), \
                patch.object(runner, "run_agent", agent):
            for index in range(2):
                attempt = self.out / str(index)
                attempt.mkdir()
                runner.run_project(project, "task", "native prompt", attempt, "model",
                                   protected=("run.py",), extra_instructions="Native tool mapping")
        self.assertEqual((project / "sql/model.sql").read_text(), "select 1")
        self.assertEqual(seen[0][1], ((project / "run.py").resolve(), "/workspace/run.py", True))
        self.assertEqual(len(seen[0]), 2)  # No source checkout or gold mount.

    @unittest.skipUnless((dacomp_de.REPO / "dacomp-de/evaluation_suite/utils.py").exists(),
                         "Run setup.sh to test upstream prompt/ID helpers")
    def test_native_dacomp_prompts_and_id_resolution(self):
        project = self.project()
        (project / "docs").mkdir()
        (project / "docs/data_contract.yaml").write_text("business_rule: KEEP_ME\n")
        (project / "question.md").write_text("Add the requested revenue metric.")
        impl = dacomp_de.native_prompt(project, "dacomp-de-impl-001")
        evol = dacomp_de.native_prompt(project, "dacomp-de-evol-001")
        self.assertIn("KEEP_ME", impl)
        self.assertIn("ref()", impl)
        self.assertIn("semicolon", impl)
        self.assertIn("Add the requested revenue metric.", evol)
        self.assertIn("modifying the `run.py` file is prohibited", evol)
        self.assertEqual(dacomp_de.config_id("task-001-suffix", {"task-001": {}}), "task-001")
        self.assertEqual(dacomp_de.config_id("task-001", {"task": {}, "task-001": {}}), "task-001")
        with self.assertRaises(ValueError):
            dacomp_de.config_id("task-001-extra", {"task": {}, "task-001": {}})

    def test_spider_score_is_native_not_exit_status(self):
        self.assertEqual(spider2_dbt.parse_score("1\n0.0 0 1\n")["score"], 0)
        self.assertEqual(spider2_dbt.parse_score("details\n1.0 1 1\n")["score"], 1)
        for text in ("", "0.5 1 2", "1.0 0 1", "No cases evaluated"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                spider2_dbt.parse_score(text)

    def test_spider_includes_execution_restrictions_actions_and_full_task(self):
        instruction = "TASK START\n" + "Required transformation.\n" * 1000 + "TASK END"
        prompt = spider2_dbt.native_prompt({"instruction": instruction})
        self.assertIn(instruction, prompt)
        self.assertIn("run `dbt run` to update the database", prompt)
        self.assertIn("must not attempt to modify the yml file", prompt)
        self.assertIn("verify the new data models generated in the database", prompt)
        self.assertIn("DON't TERMINATE with CSV FILE", prompt)
        self.assertIn("only make modifications if the SQL file clearly appears to be unfinished", prompt)
        self.assertIn("do **not** use the DuckDB CLI", prompt)
        for action in ("Bash", "Terminate", "CreateFile", "EditFile", "LOCAL_DB_SQL"):
            self.assertIn(action, prompt)

    def test_spider_preserves_prompt_profiles_and_submission_format(self):
        bench = self.root / "spider"
        project = bench / "examples/airbnb001"
        project.mkdir(parents=True)
        (project / "dbt_project.yml").write_text("profile: airbnb\n")
        (project / "profiles.yml").write_text(
            "airbnb:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: ./airbnb.duckdb\n"
        )
        (project / "airbnb.duckdb").write_bytes(b"starting database")
        (bench / "examples/spider2-dbt.jsonl").write_text(json.dumps({
            "instance_id": "airbnb001", "instruction": "Original task instruction",
        }) + "\n")
        gold = bench / "evaluation_suite/gold"
        (gold / "airbnb001").mkdir(parents=True)
        (gold / "spider2_eval.jsonl").write_text(json.dumps({
            "instance_id": "airbnb001", "evaluation": {
                "func": "duckdb_match", "parameters": {"gold": "airbnb.duckdb"},
            },
        }) + "\n")
        with patch.object(spider2_dbt, "BENCH", bench), \
                patch.object(spider2_dbt, "run_project") as agent:
            with self.assertRaises(FileNotFoundError):
                spider2_dbt.run("airbnb001", self.out, None, "duckdb")
            agent.assert_not_called()
        (gold / "airbnb001/airbnb.duckdb").write_bytes(b"gold data")

        def solve(src, task, prompt, output, model, **kwargs):
            self.assertIn("Original task instruction", prompt)
            self.assertIn("run `dbt run` to update the database", prompt)
            self.assertIn("must not attempt to modify the yml file", prompt)
            self.assertIn("actual tool calls", kwargs["extra_instructions"])
            target = output / "submission" / task
            shutil.copytree(src, target)
            return target, {"finished": True}

        with patch.object(spider2_dbt, "BENCH", bench), \
                patch.object(spider2_dbt, "run_project", solve), \
                patch.object(spider2_dbt, "container", fake_container), \
                patch.object(spider2_dbt, "command", return_value=completed("0.0 0 1\n")) as cmd:
            result = spider2_dbt.run("airbnb001", self.out, None, "duckdb")
        metadata = json.loads((self.out / "submission/results_metadata.jsonl").read_text())
        self.assertEqual(metadata, {"instance_id": "airbnb001", "answer_type": "file",
                                    "answer_or_path": "./airbnb.duckdb"})
        self.assertEqual(result["native_result"]["score"], 0)
        self.assertIn("evaluate.py", cmd.call_args.args)
        self.assertEqual((project / "airbnb.duckdb").read_bytes(), b"starting database")

    def test_dacomp_cfs_retains_partial_credit(self):
        bench = self.root / "dacomp"
        project = bench / "tasks/dacomp-de-evol-001"
        project.mkdir(parents=True)
        (project / "config").mkdir()
        for filename in ("run.py", "question.md", "config/layer_dependencies.yaml", "sample_start.duckdb"):
            (project / filename).write_text("fixture")
        suite = bench / "evaluation_suite"
        gold = suite / "gold/dacomp-de-evol-001"
        gold.mkdir(parents=True)
        (gold / "sample.duckdb").write_text("gold")
        (suite / "evaluation_config_compare.yaml").write_text(yaml.safe_dump({
            "examples": {"dacomp-de-evol-001": {"database_file": "sample.duckdb"}},
        }))
        native = {"final_score": 42.75, "partial_evaluation": {"staging": 0.5}}

        def evaluate(*args, **kwargs):
            self.assertIn("cfs", args)
            self.assertIn("--force-rebuild", args)
            runner.write_json(self.out / "evaluation/native.json", native)
            return completed()

        with patch.object(dacomp_de, "BENCH", bench), \
                patch.object(dacomp_de, "config_id", return_value="dacomp-de-evol-001"), \
                patch.object(dacomp_de, "native_prompt", return_value="native prompt"), \
                patch.object(dacomp_de, "run_project", return_value=(project, {"finished": True})) as agent, \
                patch.object(dacomp_de, "container", fake_container), \
                patch.object(dacomp_de, "command", evaluate):
            result = dacomp_de.run("dacomp-de-evol-001", self.out, None, "duckdb")
        self.assertEqual(result["native_result"], native)
        self.assertEqual(agent.call_args.kwargs["protected"], ("run.py",))

    def test_data_eng_health_failure_stops_before_agent(self):
        with patch.object(data_eng_bench, "command", return_value=completed(returncode=1)), \
                patch.object(data_eng_bench.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "healthcheck failed"):
                data_eng_bench.initialize("container", {"command": "native-setup", "retries": 2})

    def test_taskall_continues_errors_without_converting_scores(self):
        def execute(task, *args):
            if task == "broken":
                raise RuntimeError("setup failed")
            return {"status": "evaluated", "native_result": {"final_score": 0.25}}

        adapter = SimpleNamespace(REPO=Path("fake"), tasks=lambda db: ["broken", "partial"], run=execute)
        with patch.object(run, "ROOT", self.root), \
                patch.object(run.importlib, "import_module", return_value=adapter), \
                patch.object(run, "command", return_value=completed("revision\n")):
            code = run.main(["--benchmark", "dacomp-de", "--task", "taskall"])
        self.assertEqual(code, 1)
        summary = json.loads(next((self.root / "runs").rglob("summary.json")).read_text())
        self.assertEqual([row["status"] for row in summary], ["error", "evaluated"])
        self.assertEqual(summary[1]["native_result"]["final_score"], 0.25)

    def test_paths_cannot_escape_task_root(self):
        with self.assertRaises(ValueError):
            runner.within(self.root, "../gold")
        with self.assertRaises(ValueError):
            runner.within(self.root, "/gold")

    def test_shared_agent_uses_native_prompt_and_never_resets_elt_warehouse(self):
        module = runner.elt_runner()
        original_workdir = module.CONTAINER_WORKDIR
        build = module.build_tools

        def tools(name):
            self.assertEqual(module.CONTAINER_WORKDIR, "/app")
            self.assertIsNone(module._enforce_workspace("/app/model.sql"))
            self.assertIsNotNone(module._enforce_workspace("/app/symlink-to-gold"))
            self.assertIsNotNone(module._enforce_workspace("/app/run.py"))
            return build(name)

        def realpath(*args, **kwargs):
            return completed("/gold/solution.sql" if args[-1].endswith("symlink-to-gold") else args[-1])

        class Stream:
            final_output = "Finished the native task"
            context_wrapper = None

            async def stream_events(self):
                if False:
                    yield None

        with patch.object(module, "prepare_destination", side_effect=AssertionError("Unexpected ELT reset")), \
                patch.object(module, "build_tools", tools), \
                patch.object(module.Runner, "run_streamed", return_value=Stream()) as launch, \
                patch.object(runner, "command", realpath):
            result = runner.run_agent("container", "Exact original prompt", "/app", self.out,
                                      "trained-model", protected=("/app/run.py",),
                                      extra_instructions="Complete native project instructions")
        self.assertEqual(module.CONTAINER_WORKDIR, original_workdir)
        self.assertEqual(launch.call_args.kwargs["input"], "Exact original prompt")
        agent = launch.call_args.args[0]
        self.assertEqual(agent.model, "trained-model")
        self.assertNotIn("Airbyte", agent.instructions)
        self.assertNotIn("create a new", agent.instructions)
        self.assertIn("Complete native project instructions", agent.instructions)
        self.assertEqual((self.out / "system_prompt.txt").read_text(), agent.instructions)
        self.assertTrue(result["finished"])
        self.assertEqual(result["result"], "Finished the native task")

    def test_data_eng_stops_agent_before_verifier(self):
        repo = self.root / "data-eng"
        task = repo / "tasks/fix"
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "instruction.md").write_text("Original complete instructions")
        (task / "task.toml").write_text("[environment]\ncpus=1\n[agent]\n[verifier]\n")
        active = []
        commands = []

        @contextmanager
        def lifecycle(image, *args, **kwargs):
            self.assertEqual(active, [])
            name = "verifier" if ":" in image else "agent"
            active.append(name)
            yield name
            active.pop()

        def command(*args, **kwargs):
            commands.append(args)
            if "cp" in args and str(args[-1]).endswith(":/tests"):
                self.assertEqual(active, ["verifier"])
            if "verifier:/logs/verifier/." in args:
                (Path(args[-1]) / "reward.txt").write_text("0\n")
            return completed(returncode=1 if "/tests/test.sh" in args else 0)

        with patch.object(data_eng_bench, "REPO", repo), \
                patch.object(data_eng_bench, "container", lifecycle), \
                patch.object(data_eng_bench, "command", command), \
                patch.object(data_eng_bench, "run_agent", return_value={"finished": True}) as agent:
            result = data_eng_bench.run("fix", self.out, "model", "duckdb")
        self.assertEqual(agent.call_args.args[1], "Original complete instructions")
        self.assertEqual(result["native_reward"], 0)
        self.assertEqual(active, [])
        self.assertTrue(any("commit" in args for args in commands))

    def test_ade_uses_original_harness_and_custom_agent_hook(self):
        from ade_bench.harness_models import FailureMode
        repo = self.root / "ade"
        (repo / "tasks/airbnb001").mkdir(parents=True)
        (repo / "tasks/airbnb001/task.yaml").write_text("max_agent_timeout_sec: 12\n")
        logs = self.out / "logs"
        logs.mkdir()
        project_instructions = "NATIVE PROJECT POLICY\n" + "Keep this rule.\n" * 1000 + "POLICY END"
        returned = SimpleNamespace(
            results=[SimpleNamespace(failure_mode=FailureMode.NONE)],
            model_dump=lambda **kwargs: {"results": [{"is_resolved": False}]},
        )

        class Harness:
            def __init__(inner, **kwargs):
                self.assertEqual(kwargs["task_ids"], ["airbnb001.base"])
                self.assertEqual(kwargs["plugin_set_names"], ["none"])
                self.assertEqual(kwargs["n_concurrent_trials"], 1)
                inner.factory = kwargs["agent_factory"]

            def run(inner):
                agent = inner.factory.get_agent()
                result = agent.perform_task("Native ADE prompt", SimpleNamespace(
                    container=SimpleNamespace(name="prepared-by-ade")
                ), logging_dir=logs)
                self.assertEqual(result.input_tokens, 10)
                return returned

        with patch.object(ade_bench, "REPO", repo), \
                patch.object(ade_bench, "tasks", return_value=["airbnb001"]), \
                patch.object(ade_bench, "agent_defaults", return_value=SimpleNamespace(model="model", max_turns=100)), \
                patch.object(ade_bench, "command", return_value=completed(project_instructions)), \
                patch.object(ade_bench, "run_agent", return_value={
                    "finished": True, "model": "model", "num_turns": 2,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }) as run_agent, patch("ade_bench.harness.Harness", Harness):
            result = ade_bench.run("airbnb001", self.out, None, "duckdb")
        self.assertEqual(run_agent.call_args.kwargs["timeout"], 12)
        self.assertIn(project_instructions, run_agent.call_args.kwargs["extra_instructions"])
        self.assertEqual(run_agent.call_args.args[:3], ("prepared-by-ade", "Native ADE prompt", "/app"))
        self.assertEqual(result["status"], "evaluated")
        self.assertFalse(result["native_result"]["results"][0]["is_resolved"])


@unittest.skipUnless(os.environ.get("ELT_EVAL_DOCKER_TESTS") == "1", "Opt-in Docker integration")
class DockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner.build_runtime()

    def test_existing_elt_tools_and_readonly_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "run.py").write_text("original")
            module = runner.elt_runner()
            with runner.container(runner.RUNTIME_IMAGE, "/workspace", [
                (project, "/workspace", False), (project / "run.py", "/workspace/run.py", True),
            ]) as name, patch.object(module, "CONTAINER_WORKDIR", "/workspace"):
                rc, stdout, _ = asyncio.run(module._docker_exec(name, ["bash", "-lc", "pwd"]))
                self.assertEqual((rc, stdout.strip()), (0, "/workspace"))
                result = runner.command("docker", "exec", name, "bash", "-c",
                                        "printf changed > /workspace/run.py", check=False)
                self.assertNotEqual(result.returncode, 0)
                editor = module.ContainerApplyPatchEditor(name)
                asyncio.run(editor._write("/workspace/model.sql", "select 1"))
                self.assertEqual(asyncio.run(editor._read("/workspace/model.sql")), "select 1")
            self.assertEqual((project / "run.py").read_text(), "original")

    def test_original_spider_evaluator_success_and_failure(self):
        suite = runner.require(spider2_dbt.BENCH / "evaluation_suite")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gold/test").mkdir(parents=True)
            (root / "submission/test").mkdir(parents=True)
            (root / "gold/spider2_eval.jsonl").write_text(json.dumps({
                "instance_id": "test", "evaluation": {"func": "duckdb_match", "parameters": {
                    "gold": "result.duckdb", "condition_tabs": ["model"],
                    "condition_cols": [[0]], "ignore_orders": [True],
                }},
            }) + "\n")
            (root / "submission/results_metadata.jsonl").write_text(json.dumps({
                "instance_id": "test", "answer_type": "file", "answer_or_path": "result.duckdb",
            }) + "\n")
            with runner.container(runner.RUNTIME_IMAGE, "/evaluation", [
                (suite, "/evaluation", True), (root, "/fixtures", False),
            ]) as name:
                runner.command("docker", "exec", name, "python", "-c",
                               "import duckdb; "
                               "duckdb.connect('/fixtures/gold/test/result.duckdb').execute('create table model as select 1 as id'); "
                               "duckdb.connect('/fixtures/submission/test/result.duckdb').execute('create table model as select 1 as id')")
                argv = ("docker", "exec", name, "python", "evaluate.py", "--gold_dir",
                        "/fixtures/gold", "--result_dir", "/fixtures/submission")
                self.assertEqual(spider2_dbt.parse_score(runner.command(*argv).stdout)["score"], 1)
                runner.command("docker", "exec", name, "python", "-c",
                               "import duckdb; duckdb.connect('/fixtures/submission/test/result.duckdb').execute('update model set id=2')")
                self.assertEqual(spider2_dbt.parse_score(runner.command(*argv).stdout)["score"], 0)

    def test_original_dacomp_cfs_rebuild_and_partial_credit(self):
        suite = runner.require(dacomp_de.BENCH / "evaluation_suite")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gold/test").mkdir(parents=True)
            project = root / "submission/test"
            project.mkdir(parents=True)
            (root / "config.yaml").write_text(yaml.safe_dump({"examples": {"test": {
                "database_file": "result.duckdb", "layers": {"staging": {
                    "weight": 100, "tables": {"first": 1, "second": 1},
                }},
            }}}))
            script = (
                "import duckdb\n"
                "db = duckdb.connect('result.duckdb')\n"
                "db.execute('create schema staging')\n"
                "db.execute('create table staging.first as select 1 as id')\n"
                "db.execute('create table staging.second as select 2 as id')\n"
                "db.close()\n"
            )
            (project / "run.py").write_text(script)
            with runner.container(runner.RUNTIME_IMAGE, "/evaluation", [
                (suite, "/evaluation", True), (root, "/fixtures", False),
            ]) as name:
                runner.command("docker", "exec", "-w", "/fixtures/submission/test",
                               name, "python", "run.py")
                shutil.copy2(project / "result.duckdb", root / "gold/test/result.duckdb")
                argv = ("docker", "exec", name, "python", "evaluate.py", "single",
                        "--gold_dir", "/fixtures/gold", "--pred_dir", "/fixtures/submission",
                        "--example_id", "test", "--config", "/fixtures/config.yaml",
                        "--mode", "cfs", "--force-rebuild", "--output", "/fixtures/native.json")
                runner.command(*argv)
                self.assertEqual(json.loads((root / "native.json").read_text())["final_score"], 100)
                (project / "run.py").write_text(script.replace("select 2", "select 3"))
                runner.command(*argv)
                result = json.loads((root / "native.json").read_text())
                self.assertEqual(result["final_score"], 50)
                self.assertEqual(result["evaluation_mode"], "cfs")
                self.assertTrue(result["threshold_evaluation"]["run_py_success"])


if __name__ == "__main__":
    unittest.main()
