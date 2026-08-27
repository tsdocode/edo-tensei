#include <cuda.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t verify_requested = 1;
static void check(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        const char *name = "unknown";
        cuGetErrorName(result, &name);
        fprintf(stderr, "%s failed: %s (%d)\n", operation, name, result);
        exit(EXIT_FAILURE);
    }
}
static void request_verify(int signal_number) { (void)signal_number; verify_requested = 1; }
static uint64_t checksum(const unsigned char *bytes, size_t size) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) { value ^= bytes[i]; value *= 1099511628211ULL; }
    return value;
}
int main(void) {
    const char *env_mb = getenv("EDO_MODEL_MB");
    const size_t model_bytes = (size_t)(env_mb ? atoi(env_mb) : 64) * 1024 * 1024;
    CUdevice device; CUcontext context; CUdeviceptr model;
    unsigned char *weights = malloc(model_bytes); unsigned char *readback = malloc(model_bytes);
    if (!weights || !readback || model_bytes == 0) return EXIT_FAILURE;
    for (size_t i = 0; i < model_bytes; ++i) weights[i] = (unsigned char)((i * 31 + 17) & 0xff);
    const uint64_t expected = checksum(weights, model_bytes);
    check(cuInit(0), "cuInit"); check(cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cuCtxCreate(&context, 0, device), "cuCtxCreate");
    check(cuMemAlloc(&model, model_bytes), "cuMemAlloc(model)");
    check(cuMemcpyHtoD(model, weights, model_bytes), "cuMemcpyHtoD(model)");
    signal(SIGUSR1, request_verify);
    printf("model-ready pid=%d model_bytes=%zu expected_checksum=%llu\n", getpid(), model_bytes, (unsigned long long)expected);
    fflush(stdout);
    for (;;) {
        if (verify_requested) {
            check(cuMemcpyDtoH(readback, model, model_bytes), "cuMemcpyDtoH(model)");
            const uint64_t actual = checksum(readback, model_bytes);
            if (actual != expected) { fprintf(stderr, "model checksum mismatch\n"); return EXIT_FAILURE; }
            printf("gpu-model-checksum pid=%d bytes=%zu checksum=%llu\n", getpid(), model_bytes, (unsigned long long)actual);
            fflush(stdout); verify_requested = 0;
        }
        sleep(1);
    }
}
