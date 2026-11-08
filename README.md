# ELT-Eval

Setup: run `bash setup.sh` to clone ELT-Bench, Spider2, DAComp, data-eng-bench, and ADE-Bench into `source/`; existing clones are skipped.


## Install

Requires Python 3.11+ (tested with 3.12), Git, and a running Docker daemon.

```bash
bash setup.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
```

The model connection uses the original ELT-Bench Codex runner's OpenAI Agents SDK. Pass the identifier of your served trained model with `--model`; it must support the runner's shell and apply-patch tools. Omitting it uses the model configured in the cloned ELT-Bench runner. Set the SDK's `OPENAI_BASE_URL` if your model uses a compatible custom endpoint. Credentials stay in the host process.


| Benchmark | Data/environment setup |
| --- | --- |
| Spider 2.0-DBT | Follow [Spider's setup](https://github.com/xlang-ai/Spider2/tree/main/spider2-dbt): download `DBT_start_db.zip` and `dbt_gold.zip` into `source/Spider2/spider2-dbt`, then run its `python setup.py` there. This installs the starting databases and private gold files. |
| DAComp-DE | Follow [DAComp's download instructions](https://github.com/ByteDance-Seed/DAComp/tree/main/dacomp-de) for the English tasks and evaluation gold. Extract into `source/DAComp/dacomp-de/tasks/<original-task-id>/` and `evaluation_suite/gold/<config-id>/`. DE-Arch is outside this adapter's scope. |
| data-eng-bench | In `source/data-eng-bench`, run `git lfs pull`, then `docker build base-image/ -t ghcr.io/snowflake-labs/data-eng-bench-base:1.0.0`, as described in the [native setup](https://github.com/Snowflake-Labs/data-eng-bench#quickstart). The adapter builds each task's original Dockerfile on top of that image. |
| ADE-Bench | In `source/ade-bench`, run `gh release download databases --repo dbt-labs/ade-bench --pattern '*.duckdb' --dir shared/databases/duckdb`, following the [native setup](https://github.com/dbt-labs/ade-bench#4-download-the-shared-duckdb-databases). The native harness builds its task containers. |

For the two Snowflake variants, also complete the respective upstream Snowflake setup and export its `SNOWFLAKE_*` credentials. Their native initialization creates a fresh task database. Native cloud resource lifecycle rules still apply, including ADE's retained trial resources.

## Run

```bash
python run.py --benchmark spider2-dbt --task airbnb001 --model YOUR_TRAINED_MODEL
python run.py --benchmark dacomp-de --task YOUR_ORIGINAL_TASK_ID --model YOUR_TRAINED_MODEL
python run.py --benchmark data-eng-bench --task dbt-fix-division-by-zero --model YOUR_TRAINED_MODEL
python run.py --benchmark ade-bench --task airbnb001 --model YOUR_TRAINED_MODEL

# All supported tasks discovered in that benchmark's checkout/downloaded bundles:
python run.py --benchmark spider2-dbt --task taskall --model YOUR_TRAINED_MODEL

# DuckDB is the default; Snowflake is supported by data-eng-bench and ADE:
python run.py --benchmark data-eng-bench --task dbt-fix-division-by-zero --db snowflake --model YOUR_TRAINED_MODEL
```

`taskall` runs sequentially, records each result, and continues after individual errors. ADE selects ready tasks with the `base` prompt, dbt project type, and requested database. Task IDs and project filenames retain their native meanings: Spider's `airbnb001` implements transformations; ADE's `airbnb001` repairs deprecated macros.

## Adapters and agent connection

`run.py` handles the CLI, task selection, and result files. `runner.py` imports `source/ELT-Bench/agents/codex_agent/run.py` and reuses its `Agent`, streaming reasoning loop, shell tool, patch editor, defaults, and trajectory recorder. It supplies the native prompt and configurable working directory instead of invoking ELT's `main()`, Airbyte setup, new-project instructions, or warehouse reset. These changes happen at runtime; no upstream source patch is required after recloning.

Native instructions can live in more than the task description. Spider's JSONL sentence is rendered through its entire `DBT_SYSTEM` template, including its action descriptions, YAML-edit restrictions, `dbt run` requirement, verification, and database-submission rules. A separate system instruction maps the native action protocol onto ELT's tools. DAComp uses its native prompt builder, data-eng-bench passes the complete `instruction.md`, and ADE supplies both its selected prompt and the `AGENTS.md` installed by its harness. Project files remain available for inspection. The exact delivered task and system instructions are saved as `prompt.txt` and `system_prompt.txt` for each attempt.

| Adapter | Preparation and agent workspace | Native grading |
| --- | --- | --- |
| `adapter/spider2_dbt.py` | Fresh project and starting DuckDB database at `/workspace`; complete native `DBT_SYSTEM` template with the instruction from `spider2-dbt.jsonl`. Keeps profiles, YAML, macros, and packages. | Writes `results_metadata.jsonl` and calls Spider's `evaluate.py` with its original gold configuration and table comparisons. |
| `adapter/dacomp_de.py` | Fresh copy of the supplied Python/SQL pipeline at `/workspace`; uses DAComp's original prompt builder. Preserves DE-Impl's no-`ref()` instruction and mounts DE-Evol's `run.py` read-only. | Calls native `cfs` with `--force-rebuild`, which executes `python run.py`. Retains the full native score and partial-credit breakdown. |
| `adapter/data_eng_bench.py` | Original task image and healthcheck, including the starting bug; complete `instruction.md`; tools work at `/app`. Profile configuration assigned by the task remains agent work. | Transfers the finished filesystem into a fresh verifier container, injects native tests, and runs `tests/test.sh` (which invokes `test_outputs.py`). Collects its reward and logs. |
| `adapter/ade_bench.py` | Supplies an ELT agent hook to ADE's original harness. ADE copies the shared project/database, applies task setup, and supplies the selected prompt at `/app`; the hook explicitly loads its `AGENTS.md` into the agent's instructions. | ADE executes `test_setup`, loads hidden solution seeds, and runs its original tests. Retains ADE's complete result, including its all-tests-pass decision. |

Agent tools execute inside task containers. Only the task workspace is mounted for Spider/DAComp; data-eng-bench and ADE use their prepared task environments. Gold and grading files are provided during the evaluator phase. The agent never receives the host repository tree or Docker socket. File-edit roots are configurable; shell filesystem access is bounded by the task container. Original schemas and specifications remain available without conversion to ELT's `schemas/` format.


