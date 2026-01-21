# CUDA base
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Basics
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl bzip2 bash tini git \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Miniforge (Conda) ----
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) MF_ARCH="x86_64" ;; \
      arm64) MF_ARCH="aarch64" ;; \
      *) MF_ARCH="x86_64" ;; \
    esac; \
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MF_ARCH}.sh"; \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}"; \
    rm -f /tmp/miniforge.sh; \
    conda config --system --set channel_priority flexible; \
    conda clean -afy

# Use bash so `conda` hooks work
SHELL ["/bin/bash", "-lc"]

# Copy env file
COPY environment.yml /tmp/environment.yml

# Optional: override env name at build time
ARG ENV_NAME=""

# ---- Create the env (pip section in YAML is honored) ----
RUN set -eux; \
    if [[ -n "${ENV_NAME}" ]]; then \
        conda env create -n "${ENV_NAME}" -f /tmp/environment.yml; \
        ACT_ENV="${ENV_NAME}"; \
    else \
        conda env create -f /tmp/environment.yml; \
        # read 'name:' from YAML without Python deps
        ACT_ENV="$(awk '/^name:/ {print $2; exit}' /tmp/environment.yml || true)"; \
        [[ -n "${ACT_ENV}" ]] || ACT_ENV="base"; \
    fi; \
    conda clean -afy; \
    # make activation automatic for login/non-login shells
    echo 'export PATH='"${CONDA_DIR}"'/bin:$PATH' > /etc/profile.d/conda.sh; \
    echo 'eval "$(conda shell.bash hook)"' >> /etc/profile.d/conda.sh; \
    echo "conda activate ${ACT_ENV}" >> /etc/profile.d/conda.sh; \
    # also set default CMD-time activation
    echo "${ACT_ENV}" > /etc/conda_default_env

# Make it the default
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=/opt/conda/envs/${CONDA_DEFAULT_ENV}/bin:$PATH
# Defaults
WORKDIR /workspace
COPY esm2.py /workspace/esm2.py
ENTRYPOINT ["/workspace/esm2.py"]

#ENTRYPOINT ["/usr/bin/tini", "--"]
#CMD ["bash", "-lc", "EN=$(cat /etc/conda_default_env); echo Activating: $EN; conda activate \"$EN\" && python"]

