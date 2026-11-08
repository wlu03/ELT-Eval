#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p source

clone_source() {
    [ -d "source/$1/.git" ] || git clone --depth 1 --branch "$3" "$2" "source/$1"
}

clone_source "ELT-Bench" "https://github.com/uiuc-kang-lab/ELT-Bench.git" "elt_bench++"
clone_source "Spider2" "https://github.com/xlang-ai/Spider2.git" "main"
clone_source "DAComp" "https://github.com/ByteDance-Seed/DAComp.git" "main"
clone_source "data-eng-bench" "https://github.com/Snowflake-Labs/data-eng-bench.git" "master"
clone_source "ade-bench" "https://github.com/dbt-labs/ade-bench.git" "main"
