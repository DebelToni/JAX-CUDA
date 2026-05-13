FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      iproute2 \
      procps \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      ripgrep \
      tmux \
      vim \
      wget \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_DIR=/opt/venv
RUN python3 -m venv "$VENV_DIR" \
    && "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV="$VENV_DIR" \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORMS=cuda,cpu

RUN pip install --no-cache-dir --upgrade \
      "jax[cuda13]" \
      flask \
      optax \
      orbax-checkpoint \
      flax \
      chex \
      jaxtyping

RUN printf '%s\n' \
      'VIRTUAL_ENV=/opt/venv' \
      'PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
      >> /etc/environment \
    && printf '%s\n' 'source /opt/venv/bin/activate || true' > /etc/profile.d/activate-venv.sh \
    && printf '%s\n' 'set -o vi' > /etc/profile.d/vi-mode.sh \
    && printf '%s\n' 'set -g mode-keys vi' > /etc/tmux.conf

RUN curl -fsSL https://tailscale.com/install.sh | sh

ENV APP_ROOT=/workspace \
    REPO_NAME=JAX-CUDA \
    REPO_URL=https://github.com/DebelToni/JAX-CUDA \
    GIT_CLONE_DEPTH=1 \
    TS_AUTHKEY="" \
    TS_HOSTNAME="" \
    TS_TAGS="tag:gpu" \
    TS_STATE_DIR=/var/lib/tailscale \
    TS_ACCEPT_DNS=false \
    TS_ENABLE_SSH=true \
    TS_EXTRA_ARGS=""

RUN mkdir -p "$APP_ROOT" "$TS_STATE_DIR" /var/run/tailscale

COPY --chown=root:root <<'BASH' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

cd /

APP_ROOT="${APP_ROOT:-/workspace}"
REPO_NAME="${REPO_NAME:-JAX-CUDA}"
REPO_URL="${REPO_URL:-https://github.com/DebelToni/JAX-CUDA}"
GIT_CLONE_DEPTH="${GIT_CLONE_DEPTH:-1}"
REPO_DIR="${APP_ROOT}/${REPO_NAME}"

mkdir -p "$APP_ROOT"

if ! git config --system --get-all safe.directory 2>/dev/null | grep -Fxq "$REPO_DIR"; then
  git config --system --add safe.directory "$REPO_DIR" 2>/dev/null || true
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  rm -rf "$REPO_DIR" 2>/dev/null || true
  git clone --depth "$GIT_CLONE_DEPTH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --prune --depth "$GIT_CLONE_DEPTH"
  git -C "$REPO_DIR" pull --rebase --autostash --depth "$GIT_CLONE_DEPTH"
fi

if [[ -f "$REPO_DIR/pyproject.toml" || -f "$REPO_DIR/setup.py" ]]; then
  /opt/venv/bin/pip install -e "$REPO_DIR"
fi

mkdir -p "$TS_STATE_DIR" /var/run/tailscale

nohup /usr/sbin/tailscaled \
  --tun=userspace-networking \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &

for _ in {1..100}; do
  [[ -S /var/run/tailscale/tailscaled.sock ]] && break
  sleep 0.1
done

hn="${TS_HOSTNAME:-gpu-box}"
UP_ARGS=(--hostname="$hn" --accept-dns="${TS_ACCEPT_DNS:-false}" --advertise-tags="${TS_TAGS:-tag:gpu}")

if [[ "${TS_ENABLE_SSH:-true}" == "true" ]]; then
  UP_ARGS+=(--ssh)
fi

if [[ -n "${TS_AUTHKEY:-}" ]]; then
  UP_ARGS+=(--authkey="$TS_AUTHKEY")
fi

EXTRA=()
if [[ -n "${TS_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<<"${TS_EXTRA_ARGS}"
fi

for _ in {1..60}; do
  if tailscale up "${UP_ARGS[@]}" "${EXTRA[@]}"; then
    break
  fi
  sleep 2
done

tailscale status || true

if [[ $# -gt 0 ]]; then
  exec "$@"
else
  exec sleep infinity
fi
BASH

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
