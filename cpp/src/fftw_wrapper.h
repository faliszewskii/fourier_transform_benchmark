#pragma once
#include <fftw3.h>

class FftwWrapper {
    using Vt = double;
    using Complex = std::complex<Vt>;
public:
    FftwWrapper(std::span<const Complex> input, std::span<Complex> output): _input(input), _output(output) {
        const size_t N = _input.size();
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