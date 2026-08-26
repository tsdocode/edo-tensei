# Native CUDA checkpoint fixture

This fixture creates a CUDA driver context, allocates 4 KiB of device memory,
writes a deterministic pattern, and verifies that pattern on `SIGUSR1`. It is
used to validate the CUDA checkpoint API and combined CUDA+CRIU resurrection.

Build it on the development host:

```bash
gcc -I/usr/local/cuda/include \
  examples/cuda-checkpoint/cuda-fixture.c \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lcuda -o examples/cuda-checkpoint/cuda-fixture
```

Run the combined validation from the repository root:

```bash
./examples/cuda-checkpoint/run-combined-demo.sh
```

The fixture is quiescent between verification cycles. A production framework
integration must provide its own request-draining/quiescence hook before
calling `edo freeze`; Edo cannot infer in-flight application work from an
arbitrary process.
