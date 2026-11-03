#!/bin/bash
set -e

JAX_VERSION="0.7.1"
NVIDIA_DIST="/usr/local/lib/python3.12/dist-packages/nvidia"

# Install CUDA extras only on first run (or after pod terminate+recreate).
# On stop+restart the container disk persists so this is skipped.
if [ ! -d "$NVIDIA_DIST" ]; then
    echo "[startup] Installing jax[cuda12]==${JAX_VERSION} CUDA packages (~3 GB)..."
    mkdir -p /workspace/tmp
    TMPDIR=/workspace/tmp pip3 install --no-cache-dir "jax[cuda12]==${JAX_VERSION}"
    echo "[startup] Done."
fi

# ldconfig must run every container start (its cache is on the ephemeral disk).
echo "[startup] Registering CUDA libraries..."
find "$NVIDIA_DIST" -name 'lib' -type d \
    | while read p; do echo "$p"; done > /etc/ld.so.conf.d/nvidia-pip.conf
ldconfig

echo "[startup] JAX devices: $(python3 -c 'import jax; print(jax.devices())')"

exec sleep infinity
