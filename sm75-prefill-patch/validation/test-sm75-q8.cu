#include "sm75-q8.cuh"
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; \
} } while (0)

static __device__ bool equal_half_bits(half a, half b) {
    const unsigned x = __half_as_ushort(a), y = __half_as_ushort(b);
    // Accept either NaN payload/sign, but not a NaN versus a non-NaN.
    const bool x_nan = (x & 0x7c00u) == 0x7c00u && (x & 0x03ffu) != 0;
    const bool y_nan = (y & 0x7c00u) == 0x7c00u && (y & 0x03ffu) != 0;
    return x == y || (x_nan && y_nan);
}

static __global__ void check_pairs(unsigned * failures) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
    const unsigned pair = blockIdx.x*blockDim.x + threadIdx.x;
    if (pair >= 65536) {
        return;
    }
    const unsigned ua = pair & 255, ub = pair >> 8;
    const int a = ua < 128 ? int(ua) : int(ua) - 256;
    const int b = ub < 128 ? int(ub) : int(ub) - 256;
    const half2 fast = ggml_cuda_sm75_q8_pair_to_half2(int8_t(a), int8_t(b));
    const half2 ref = __halves2half2(__int2half_rn(a), __int2half_rn(b));
    const unsigned short scales[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
        0x0800, 0x8800, 0x2000, 0xa000, 0x3400, 0xb400, 0x3555, 0xb555,
        0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x7bff, 0xfbff,
        0x7c00, 0xfc00, 0x7e00
    };
    for (unsigned short bits : scales) {
        const half2 d = __half2half2(__ushort_as_half(bits));
        const half2 x = __hmul2(d, fast), y = __hmul2(d, ref);
        if (!equal_half_bits(__low2half(x), __low2half(y)) ||
            !equal_half_bits(__high2half(x), __high2half(y))) {
            atomicAdd(failures, 1u);
        }
    }
#else
    // A wrongly targeted test must never silently report equivalence.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(failures, 1u);
    }
#endif
}

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::fprintf(stderr, "SKIP: a working CUDA device is required\n");
        return 77;
    }
    int device = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp props;
        CUDA_OK(cudaGetDeviceProperties(&props, i));
        if (props.major == 7 && props.minor == 5) {
            device = i;
            break;
        }
    }
    if (device < 0) {
        std::fprintf(stderr, "SKIP: an SM75 GPU is required\n");
        return 77;
    }
    CUDA_OK(cudaSetDevice(device));
    unsigned * failures = nullptr;
    CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&failures), sizeof(unsigned)));
    CUDA_OK(cudaMemset(failures, 0, sizeof(unsigned)));
    check_pairs<<<256, 256>>>(failures);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    unsigned result = 0;
    CUDA_OK(cudaMemcpy(&result, failures, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(failures));
    if (result != 0) {
        std::fprintf(stderr, "FAIL: %u mismatches\n", result);
        return 1;
    }
    std::printf("PASS: device %d, 65536 INT8 pairs x 27 FP16 scales\n", device);
    return 0;
}
