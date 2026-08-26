# Phase 0 Environment Record

Recorded after the initial development setup.

## Host capabilities

- OS: Ubuntu 24.04
- Architecture: x86_64
- GPU: NVIDIA H100 80GB HBM3
- NVIDIA driver: 570.211.01
- CUDA toolkit: 12.8 (`/usr/local/cuda`)
- CUDA compiler: `/usr/local/cuda/bin/nvcc`
- CUDA driver library: `/lib/x86_64-linux-gnu/libcuda.so.1`
- CUDA checkpoint symbols: present
- CRIU: 4.2.1, built from the official release source

## Validation

- `nvidia-smi`: passed
- `nvcc --version`: passed
- `criu --version`: passed
- `sudo criu check`: passed with `Looks good.`

CRIU was built from source because this Ubuntu 24.04 host had no APT candidate for the package. The build completed with the optional nftables support warning; basic CRIU kernel checks pass.

## Shell setup

Source `env.sh` from the project directory to add the existing CUDA toolkit compiler and libraries to the current shell.

## Remaining Phase 0 / Phase 1 work

- Select and add the project license file.
- Remove the scaffold dead-code warning from the typed error enum.
- Implement real CRIU and CUDA capability checks in `edo doctor`.
