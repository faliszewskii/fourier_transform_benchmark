#pragma once

namespace CudaCT {

template <typename Vt>
struct Complex {
    Vt real;
    Vt imag;
};

template <typename Vt>
__host__ __device__ Complex<Vt> operator+(Complex<Vt> a, Complex<Vt> b) {
    return {a.real + b.real, a.imag + b.imag};
}

template <typename Vt>
__host__ __device__ Complex<Vt> operator-(Complex<Vt> a, Complex<Vt> b) {
    return {a.real - b.real, a.imag - b.imag};
}

template <typename Vt>
__host__ __device__ Complex<Vt> operator*(Complex<Vt> a, Complex<Vt> b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

template <typename Vt>
__global__ void calculateButterfly(Complex<Vt>* data, Complex<Vt>* weights, size_t N,
    size_t groupCount, size_t groupSize, size_t jump);

template <typename Vt>
__global__ void calculateWeights(Complex<Vt>* weights, size_t N);

template <typename Vt>
__global__ void mirrorBits(Complex<Vt>* data, size_t N, int log2N);

template <typename Vt>
__host__ void setup(Complex<Vt>* weights, size_t N);

template <typename Vt>
__host__ void execute(Complex<Vt>* data, Complex<Vt>* weights, size_t N);

}