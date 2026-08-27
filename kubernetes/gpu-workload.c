#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    CUdevice device;
    CUcontext context;
    CUdeviceptr buffer;
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) return 10;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 11;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return 12;
    if (cuMemAlloc(&buffer, 256ULL * 1024ULL * 1024ULL) != CUDA_SUCCESS) return 13;
    if (cuMemsetD8(buffer, 0x5a, 256ULL * 1024ULL * 1024ULL) != CUDA_SUCCESS) return 14;
    printf("gpu-workload ready pid=%d allocation=256MiB\n", getpid());
    fflush(stdout);
    for (;;) {
        sleep(1);
    }
}
