# Pinned environment for medchem.
#
# NOTE ON REPRODUCIBILITY: the frozen results were produced on Python 3.14.6 (recorded per run in
# provenance/*/run_manifest.json under `software`), not on the version below. This image is a
# convenience for running the pipeline, NOT the environment the published metrics came from -- use
# `uv sync --frozen` on 3.14 to reproduce those exactly.
FROM python:3.14-slim

# uv for fast, locked installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

# Build strictly from the committed lockfile so the image is the pinned env.
# Core + science + dev so the image actually runs the pipeline (the GPU generative/structure
# extras get a separate CUDA image when those stages come online).
RUN uv sync --frozen --extra science --extra dev

ENTRYPOINT ["uv", "run", "medchem"]
CMD ["--help"]
