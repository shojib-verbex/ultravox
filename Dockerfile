# CUDA Version
ARG CUDA_VERSION=12.8.1
ARG BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04"

FROM ${BASE_IMAGE}

ENV \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    CUDA_VERSION=${CUDA_VERSION} \
    POETRY_HOME=/opt/poetry \
    PROJECT_ROOT=/app \
    VENV_PATH=/app/.venv \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    HF_HOME=/cache/huggingface

# Set working directory
WORKDIR ${PROJECT_ROOT}

# Install Python 3.11, venv, and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common build-essential git vim openssh-client emacs \
    curl wget ffmpeg libavcodec-extra \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Install just
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Install poetry in its own venv
RUN python3.11 -m venv $POETRY_HOME
ENV PATH="${VENV_PATH}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${POETRY_HOME}/bin"
RUN $POETRY_HOME/bin/pip install poetry==$POETRY_VERSION

# Copy poetry files first (for caching)
COPY pyproject.toml poetry.lock $PROJECT_ROOT/

# Install dependencies (without dev dependencies for smaller image)
# Use --with dev if you need development tools
RUN poetry install --no-interaction --no-ansi --no-cache --no-root && \
    rm -rf ~/.cache/pypoetry

# Copy project files
COPY . $PROJECT_ROOT/

# Install the project itself
RUN poetry install --no-interaction --no-ansi --no-cache --only-root

# Create cache directories
RUN mkdir -p /cache/huggingface /app/output

# Default command
CMD ["bash"]
