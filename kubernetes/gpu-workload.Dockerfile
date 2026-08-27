FROM nvidia/cuda:12.8.1-base-ubuntu24.04
COPY gpu-workload /usr/local/bin/gpu-workload
ENTRYPOINT ["/usr/local/bin/gpu-workload"]
