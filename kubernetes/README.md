# Edo on Kubernetes

This is the first Kubernetes MVP for container-aware GPU snapshots. It uses a
privileged node-local agent and enters the target container's namespaces before
calling Edo. This keeps the container root filesystem, `/dev/shm`, sockets, and
mount namespace together with the CRIU images.

## Build and install

### Local k3s GPU test

The development machine was validated with k3s `v1.36.3+k3s1` and the
NVIDIA device plugin `v0.19.3`. Install k3s without Traefik, then configure
the device plugin to use the NVIDIA runtime:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC='server --disable=traefik --write-kubeconfig-mode=640' sh -
sudo k3s kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.19.3/deployments/static/nvidia-device-plugin.yml
sudo k3s kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset \
  --type='strategic' -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'
```

Verify that the node advertises a GPU and run the checked-in smoke test:

```bash
sudo k3s kubectl get nodes -o wide
sudo k3s kubectl apply -f kubernetes/gpu-smoke-test.yaml
sudo k3s kubectl -n edo-system wait --for=jsonpath='{.status.phase}'=Succeeded pod/edo-gpu-smoke --timeout=180s
sudo k3s kubectl -n edo-system logs pod/edo-gpu-smoke
```

On the validated host this reports an NVIDIA H100 80GB with driver 570.211.01
and CUDA 12.8. The smoke test is only a GPU/runtime check; it does not yet
checkpoint a Kubernetes workload.

### Real GPU workload test

`gpu-workload-pod.yaml` is a long-lived CUDA Driver API process. It allocates
256 MiB on the GPU and mounts the Edo/CRIU binaries and snapshot directory into
the target namespace. Build/import it locally for k3s, then apply the Pod:

```bash
gcc -O2 -I/usr/local/cuda-12.8/include kubernetes/gpu-workload.c \
  -L/usr/lib/x86_64-linux-gnu -lcuda -o kubernetes/gpu-workload
docker build -f kubernetes/gpu-workload.Dockerfile -t edo-gpu-workload:test kubernetes
docker save edo-snapshot-agent:test edo-gpu-workload:test | sudo k3s ctr images import -
sudo k3s kubectl apply -f kubernetes/gpu-workload-pod.yaml
```

The validated k3s run uses CRIU live-checkpoint mode and a placeholder Pod on
restore. Dump reached CUDA `CHECKPOINTED`, created the manifest, and resumed
the source process. After replacing the source with a PID-1-only placeholder,
restore recreated the CUDA process and unlocked it successfully. The measured
restore was 17.35s for this minimal 256 MiB fixture. Instrumented timing shows
that this was mostly integrity hashing, not CUDA: `cuInit()` is about 11ms and
CRIU restore is about 0.46s. A clean verification-on run measured about 16.4s
hashing the 832 MiB pages image. Using the optimized native SHA-256 path,
snapshot creation completed in 2.11s with the pages hash taking about 0.7s.

The required Kubernetes behavior is therefore proven for this fixture. Model
servers still need a controller to create the placeholder, map container PIDs,
and run health/inference readiness checks before advertising service readiness.

### Real vLLM container test

The checked-in `vllm-qwen3-pod.yaml` runs the local `vllm-local:cuda12.8`
container with Qwen3-0.6B on the H100. The real server reached `/health`,
loaded its model, completed torch compilation and CUDA graph capture, and
served a completion (`2+2 is 4.`). Cold Pod-to-API readiness was about 33.3 s.

The live group snapshot of the API parent plus `VLLM::EngineCore` succeeded
with the port-io-uring CRIU fork. CRIU logged three io_uring ring images, and
the snapshot was 9.2 GiB. Snapshot end-to-end time was 44.8 s, including 33.8
s for integrity hashing of an 8.7 GiB pages image. A pre-restore request took
74 ms.

The validated CNI restore path joins the destination Pod's network namespace
and remaps source Pod IPv4 endpoints, including IPv4-mapped IPv6 TCP sockets,
to the destination Pod IP. Shared vLLM/Triton cache volumes preserve rebuildable
JIT artifacts required by restored mappings. The Qwen3-0.6B CNI test restored
in 6.5s end to end (CRIU 4.9s, CUDA restore/unlock 1.6s); `/health` returned
200 and a real completion returned valid output in 41ms. The snapshot agent
excludes runtime-owned mounts and records the source endpoint in `network.json`.
The remaining production work is controller-managed Pod/CRI discovery,
metadata lifecycle, cache-volume provisioning, and readiness gating.

Build the agent image and publish it to a registry reachable by every GPU node:

```bash
docker build -t ghcr.io/tsdocode/edo-snapshot-agent:dev kubernetes/snapshot-agent
docker push ghcr.io/tsdocode/edo-snapshot-agent:dev
kubectl apply -f kubernetes/namespace-rbac.yaml
kubectl apply -f kubernetes/edo-snapshot-crd.yaml
kubectl apply -f kubernetes/snapshot-agent/service.yaml
```

Each node must provide `/opt/edo/bin/edo` and `/opt/edo/bin/criu` (the Edo
release binary and the CUDA/io_uring CRIU fork), plus `cuda-checkpoint` in the
host PATH. The target Pod must request an exclusive NVIDIA GPU and use the
same driver/GPU compatibility policy as the snapshot.

The target Pod must mount the node's snapshot directory at the same path (for
example, `/var/lib/edo-snapshots`) so that the namespace-entered Edo process
can see the CRIU images. In production this should be a node-local encrypted
volume or a CSI volume with a strict access policy.

The agent contains an optional `EDO_CUDA_PREWARM` diagnostic hook. It is
disabled by default because testing shows that CUDA initialization state is
process-local for this driver path; the actual restore bottleneck was snapshot
integrity hashing. CRIU's host dependencies are kept under `/opt/edo/lib` and
the CRIU binary uses an embedded relative runpath; do not set a global
`LD_LIBRARY_PATH`, because that can override the target image's glibc. Driver
and GPU compatibility must still be enforced by the deployment policy.

## Agent API

The MVP deliberately takes PIDs; a controller can resolve Pod/container IDs
through the CRI runtime and perform application quiescing/readiness checks.
The agent enters the target namespaces and translates host PIDs to inner PIDs:

```bash
curl -X POST http://NODE:8787/v1/snapshot \
  -H content-type:application/json \
  -d '{"name":"triton-v1","host_pid":1234,"cuda_pids":[1234,1278]}'
```

For restore, create a placeholder Pod with the same image, mounts, GPU
request, and namespace layout. Keep its target process alive long enough for
the agent to enter its namespaces, then call:

```bash
curl -X POST http://NODE:8787/v1/restore \
  -H content-type:application/json \
  -d '{"name":"triton-v1","host_pid":4567}'
```

For a trusted node-local snapshot, `skip_integrity: true` avoids re-hashing
large memory pages on the serving path. The default remains verification-on;
only use this option when the snapshot directory is protected and its content
was authenticated at creation time.

The workload controller must only report Ready after CUDA restore/unlock and a
model-server health/inference probe. Same-node restore is the supported MVP;
cross-node restore additionally needs identical GPU topology, driver/CUDA
compatibility, and shared checkpoint storage. Checkpoint artifacts contain
process memory and must be root-only and encrypted at rest.

This agent is intentionally not a general-purpose operator yet. Pod lookup,
quiesce hooks, object-storage upload, retry state, and admission/readiness
integration are the next layer.
