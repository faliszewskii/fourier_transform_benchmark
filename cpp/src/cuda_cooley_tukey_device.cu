#include <cuda_runtime.h>
#include <math_constants.h>
#include "cuda_cooley_tukey_device.cuh"

namespace CudaCT {

template<typename Vt>
__global__ void calculateButterfly(Complex<Vt> *data, Complex<Vt> *weights, size_t N,
        size_t groupCount, size_t groupSize, size_t jump) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N/2) return;

    size_t group = tid / (groupSize/2);
    size_t node = tid % (groupSize/2);
    size_t k = group * groupSize + node;
    Complex u = data[k];
    Complex t = weights[node*groupCount] * data[k + jump];
    data[k] = u + t;
    data[k + jump] = u - t;
}

template <typename Vt>
__global__ void calculateWeights(Complex<Vt>* weights, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N/2) return;

    constexpr Vt pi = std::is_same_v<Vt, double>? CUDART_PI: CUDART_PI_F;
    auto angle = -2 * pi * tid / N;
    weights[tid] =  Complex<Vt>(cos(angle), sin(angle));
}

template <typename Vt>
__global__ void mirrorBits(Complex<Vt>* data, size_t N, int log2N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    size_t r = 0;
    size_t n = tid;
    for (int i = 0; i < log2N; ++i) {
        r <<= 1;
        r |= n & 1;
        n >>= 1;
    }

    // Bit mirroring relation is symmetric, perform exchange only once to prevent data race.
    if (r > tid) {
        auto tmp = data[tid];
        data[tid] = data[r];
        data[r] = tmp;
    }
}

template <typename Vt>
__host__ void execute(Complex<Vt>* data, Complex<Vt>* weights, size_t N) {
    const int log2N = std::bit_width(N) - 1;

    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int blocksHalf = (N/2 + threads - 1) / threads;

    mirrorBits<<<blocks, threads>>>(data, N, log2N);

    const int iterations = log2N;
    for (int it = 0; it < iterations; ++it) {
        const size_t groupCount = 1 << (iterations - it - 1);
        const size_t groupSize = N / groupCount;
        const size_t jump = groupSize / 2;
        calculateButterfly<<<blocksHalf, threads>>>(data, weights, N, groupCount, groupSize, jump);
        cudaDeviceSynchronize();
    }
}

template void execute<float>(Complex<float>*, Complex<float>*, size_t);
template void execute<double>(Complex<double>*, Complex<double>*, size_t);

template <typename Vt>
__host__ void setup(Complex<Vt>* weights, size_t N) {
    size_t weightN = N / 2;
    constexpr int threads = 256;
    const int blocks = (weightN + threads - 1) / threads;

    calculateWeights<<<blocks, threads>>>(weights, N);
}

template void setup<float>(Complex<float>*, size_t);
template void setup<double>(Complex<double>*, size_t);

}