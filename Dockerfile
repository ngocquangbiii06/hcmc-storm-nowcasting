FROM mambaorg/micromamba:1.5-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
COPY --chown=$MAMBA_USER:$MAMBA_USER configs ./configs

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir -e ".[orchestration]"

# data/ is mounted from the host:  docker run -v ${PWD}/data:/app/data stormcast run-day --start 2025-08-01 --end 2025-08-01
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "stormcast"]
CMD ["--help"]
