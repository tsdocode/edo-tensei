#include <cuda.h>
#include <signal.h>
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

static void request_verify(int signal_number) {
    (void)signal_number;
    verify_requested = 1;
}

int main(void) {
    CUdevice device;
    CUcontext context;
    CUdeviceptr allocation;
    unsigned char pattern[4096];
    unsigned char readback[4096];

    for (size_t i = 0; i < sizeof(pattern); ++i) {
        pattern[i] = (unsigned char)(i & 0xff);
    }

    check(cuInit(0), "cuInit");
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cuCtxCreate(&context, 0, device), "cuCtxCreate");
    check(cuMemAlloc(&allocation, sizeof(pattern)), "cuMemAlloc");
    check(cuMemcpyHtoD(allocation, pattern, sizeof(pattern)), "cuMemcpyHtoD");

    signal(SIGUSR1, request_verify);
    printf("cuda-fixture pid=%d allocation=%llu\n", getpid(),
           (unsigned long long)allocation);
    fflush(stdout);

    for (;;) {
        if (verify_requested) {
            unsigned long checksum = 0;
            check(cuMemcpyDtoH(readback, allocation, sizeof(readback)), "cuMemcpyDtoH");
            for (size_t i = 0; i < sizeof(readback); ++i) {
                checksum = (checksum + readback[i]) & 0xffffffffUL;
                if (readback[i] != pattern[i]) {
                    fprintf(stderr, "GPU pattern mismatch at byte %zu\n", i);
                    return EXIT_FAILURE;
                }
            }
            printf("gpu-pattern-ok checksum=%lu\n", checksum);
            fflush(stdout);
            verify_requested = 0;
        }
        sleep(1);
    }
}
