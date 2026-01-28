#pragma once
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <span>
#include <complex>
#include <cassert>

template<ft_data_type Vt>
class CuFFTWrapper {
    using cufftVt = std::conditional_t<std::is_same_v<Vt, double>, cufftDoubleComplex, cufftComplex>;
    using Complex = std::complex<Vt>;
public:
    CuFFTWrapper(std::span<const Complex> input, std::span<Complex> output): _input(input), _output(output) {
        const size_t N = _input.size();
        constexpr auto cufftType = std::is_same_v<Vt, float>? CUFFT_C2C : CUFFT_Z2Z;
        cufftPlan1d(&_plan, N, cufftType, 1);
    }

    void execute() {
        const size_t N = _input.size();
        auto *h_in = reinterpret_cast<cufftVt*>(const_cast<Complex*>(_input.data()));
        auto *h_out = reinterpret_cast<cufftVt*>(_output.data());

        cufftVt* d_in = nullptr;
        cudaMalloc(&d_in, sizeof(cufftVt) * N);
        cudaMemcpy(d_in, h_in, sizeof(cufftVt) * N, cudaMemcpyHostToDevice);

        if constexpr (std::is_same_v<Vt, float>) {
            cufftExecC2C(_plan, d_in, d_in, CUFFT_FORWARD);
        }
        else {
            cufftExecZ2Z(_plan, d_in, d_in, CUFFT_FORWARD);
        }

        cudaMemcpy(h_out, d_in, sizeof(cufftVt) * N, cudaMemcpyDeviceToHost);
        cudaFree(d_in);
    }

    ~CuFFTWrapper() {
        cufftDestroy(_plan);
    }
private:
    std::span<const Complex> _input;
    std::span<Complex> _output;
    cufftHandle _plan{};
};