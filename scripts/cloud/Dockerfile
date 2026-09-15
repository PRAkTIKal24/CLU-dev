# syntax=docker/dockerfile:1
#
# CHLU cloud runtime image.
#
# Mirrors the CSF3 environment recipe (scripts/csf3/README.md §3.1) exactly:
#   * `uv sync --frozen --extra cuda` -> the laptop's uv.lock pins, no resolver
#   * JAX CUDA support comes from SELF-CONTAINED pip wheels
#     (jax[cuda12] -> jax-cuda12-plugin[with-cuda], which bundle CUDA + cuDNN)
#   * therefore the base image deliberately carries NO CUDA toolkit. A CUDA on
#     LD_LIBRARY_PATH shadows the bundled libs and breaks jax -- the same trap
#     `module load libs/cuda/...` sets on CSF3. The host contributes the NVIDIA
#     DRIVER only (>= 525.60.13; the GCE DLVM boot image ships 580.x).
#
# Both FROM/COPY --from sources are pinned BY DIGEST, never by tag: a floating
# tag makes a re-run silently irreproducible and we would never know.
#   python:3.11-slim-bookworm  resolved 2026-09-15
#   ghcr.io/astral-sh/uv:0.9.7 resolved 2026-09-15

FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

# uv, copied from its own digest-pinned image (no curl|sh into the build).
COPY --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# --- Cloud Storage client ----------------------------------------------------
# The payload checkpoints to GCS from INSIDE the container, so the container
# needs a GCS client of its own. Deliberately NOT a new Python dependency:
# pyproject.toml/uv.lock are owned by the CSF3 path and adding a package there
# would change the environment on a cluster that does not need it.
#
# Pinned to CLI 585.0.0 -- the same version the laptop drives the project with
# -- by VERSIONED URL and SHA-256, not by "latest". The bundled CPython is
# dropped in favour of the image's 3.11 (CLOUDSDK_PYTHON), which is what saves
# most of the footprint.
ARG GCLOUD_VERSION=585.0.0
ARG GCLOUD_SHA256=7b97198ef306f5400b67f057f7415a46bd9a34367eeabd87516ee3f74bc76a36
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fsSL -o /tmp/gcloud.tgz \
      "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-x86_64.tar.gz"; \
    echo "${GCLOUD_SHA256}  /tmp/gcloud.tgz" | sha256sum -c -; \
    tar -xzf /tmp/gcloud.tgz -C /opt; \
    rm -rf /tmp/gcloud.tgz \
           /opt/google-cloud-sdk/platform/bundledpythonunix \
           /opt/google-cloud-sdk/bin/anthoscli \
           /opt/google-cloud-sdk/platform/gsutil_py2 \
           /opt/google-cloud-sdk/.install/.backup; \
    apt-get purge -y curl; apt-get autoremove -y; rm -rf /var/lib/apt/lists/*
ENV CLOUDSDK_PYTHON=/usr/local/bin/python3 \
    CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
    PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /opt/chlu

# Dependency layer first: it is ~6 GB of CUDA wheels and must not be
# invalidated by a source edit.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra cuda --no-install-project

# Source layer.
COPY chlu ./chlu
COPY scripts/cloud ./scripts/cloud
RUN uv sync --frozen --no-dev --extra cuda

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Persistent XLA compilation cache (mirrors env.sh §3.4). Bind-mounted to a
# host path by startup.sh so it survives container restarts on the same VM.
ENV JAX_COMPILATION_CACHE_DIR=/var/chlu/jax_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

# Build-time record, readable at runtime: `docker inspect` / `chlu-image-info`.
ARG GIT_COMMIT=unknown
ARG BUILD_ID=unknown
ENV CHLU_IMAGE_GIT_COMMIT=${GIT_COMMIT} \
    CHLU_IMAGE_BUILD_ID=${BUILD_ID}
LABEL org.opencontainers.image.revision=${GIT_COMMIT} \
      org.opencontainers.image.source="https://github.com/forgis/CHLU" \
      org.opencontainers.image.description="CHLU CUDA runtime (uv --frozen --extra cuda, bundled CUDA wheels)"

# CPU-side import check: fails the build if the lock did not produce a usable
# jax. (A GPU assert cannot run here -- the builder has no GPU; startup.sh
# preflights the device on the real VM.)
RUN python -c "import jax, chlu; print('image OK: jax', jax.__version__, '| chlu', chlu.__version__ if hasattr(chlu,'__version__') else '?')" \
 && gcloud --version | head -2

CMD ["python", "/opt/chlu/scripts/cloud/payload_smoke.py", "--help"]
