FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    duckdb dbt-duckdb pandas numpy PyYAML tqdm google-cloud-bigquery regex

WORKDIR /workspace
