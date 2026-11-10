# featurelab - reproducible runtime for the whole team.
#
#   docker compose build
#   docker compose run --rm featurelab controls
#
# The container is also the real security boundary for LLM-generated code
# (see src/featurelab/features/gate.py for the other two layers).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FEATURELAB_DATA=/data \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app
# dependencies first: this layer is cached until pyproject.toml changes
COPY pyproject.toml README.md ./
RUN mkdir -p src/featurelab && echo '__version__ = "0"' > src/featurelab/__init__.py \
    && pip install ".[dev]" && pip uninstall -y featurelab
COPY src ./src
COPY tests ./tests
RUN pip install --no-deps .

# run as an unprivileged user; /data is the only writable mount that matters
RUN useradd --create-home --uid 1000 lab && mkdir -p /data /work && chown -R lab /data /work
USER lab
WORKDIR /work
VOLUME ["/data"]
ENTRYPOINT ["featurelab"]
CMD ["--help"]
