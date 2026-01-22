#pragma once
#include <fftw3.h>
#include "fourier_transform.h"

template<ft_data_type Vt>
class FftwWrapper;

template<>
class FftwWrapper<float> {
    using Complex = std::complex<float>;
public:
    FftwWrapper(std::span<const Complex> input, std::span<Complex> output): _input(input), _output(output) {
        const size_t N = _input.size();
        // reinterpret_cast compliant with https://fftw.org/fftw3.pdf ch.4.1.1
        auto *in = reinterpret_cast<fftwf_complex *>(const_cast<Complex*>(_input.data()));
        auto *out = reinterpret_cast<fftwf_complex *>(_output.data());
        _plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    }

    void execute() {
        fftwf_execute(_plan);
    }

    ~FftwWrapper() {
        fftwf_destroy_plan(_plan);
    }

private:
    std::span<const Complex> _input;
    std::span<Complex> _output;
    fftwf_plan _plan;
};

template<>
class FftwWrapper<double> {
    using Complex = std::complex<double>;
public:
    FftwWrapper(std::span<const Complex> input, std::span<Complex> output): _input(input), _output(output) {
        const size_t N = _input.size();
        // reinterpret_cast compliant with https://fftw.org/fftw3.pdf ch.4.1.1
        auto *in = reinterpret_cast<fftw_complex *>(const_cast<Complex*>(_input.data()));
        auto *out = reinterpret_cast<fftw_complex *>(_output.data());
        _plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    }

    void execute() {
        fftw_execute(_plan);
    }

    ~FftwWrapper() {
        fftw_destroy_plan(_plan);
    }

private:
    std::span<const Complex> _input;
    std::span<Complex> _output;
    fftw_plan _plan;
};