#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include "../fourier_transform.h"
#include "cuda_cooley_tukey_device.cuh"

template<ft_data_type Vt>
class CudaCooleyTukeyFFT {
    using Complex = CudaCT::Complex<Vt>;
public:
    CudaCooleyTukeyFFT(std::span<const std::complex<Vt>> input, std::span<std::complex<Vt>> output): _input(input), _output(output) {
        const size_t N = _input.size();
        assert((N & N-1) == 0); // Assert that N is power of 2.

        cudaMalloc(&_d_weights, sizeof(Complex) * N/2);
        CudaCT::setup(_d_weights, N);
    }

    void execute() {
        const size_t N = _input.size();
        auto *h_in = reinterpret_cast<Complex*>(const_cast<std::complex<Vt>*>(_input.data()));
        auto *h_out = reinterpret_cast<Complex*>(_output.data());

        Complex* d_in = nullptr;
        cudaMalloc(&d_in, sizeof(Complex) * N);
        cudaMemcpy(d_in, h_in, sizeof(Complex) * N, cudaMemcpyHostToDevice);

        CudaCT::execute<Vt>(d_in, _d_weights, N);

        cudaMemcpy(h_out, d_in, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
        cudaFree(d_in);
    }

    ~CudaCooleyTukeyFFT() {
        cudaFree(_d_weights);
    }
private:
    std::span<const std::complex<Vt>> _input;
    std::span<std::complex<Vt>> _output;

    Complex* _d_weights;
};