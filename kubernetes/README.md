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

The workload controller must only report Ready after CUDA restore/unlock and a
model-server health/inference probe. Same-node restore is the supported MVP;
cross-node restore additionally needs identical GPU topology, driver/CUDA
compatibility, and shared checkpoint storage. Checkpoint artifacts contain
process memory and must be root-only and encrypted at rest.

This agent is intentionally not a general-purpose operator yet. Pod lookup,
quiesce hooks, object-storage upload, retry state, and admission/readiness
integration are the next layer.
