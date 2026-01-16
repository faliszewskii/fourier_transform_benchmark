#pragma once
#include <complex>
#include <ranges>

template<std::floating_point Vt>
class NaiveDFT {
    using Complex = std::complex<Vt>;
    static constexpr Vt pi = std::numbers::pi_v<Vt>;
public:
    NaiveDFT(std::span<const Complex> input, std::span<Complex> output): _input(input), _output(output) {}

    void execute() {
        const size_t N = _input.size();

        std::ranges::fill(_output, 0);
        for (size_t k = 0; k < N; ++k) {
            for (size_t n = 0; n < N; ++n) {
                auto angle = -2 * pi * n / N * k;
                _output[k] += _input[n] * Complex(cos(angle), sin(angle));
            }
        }
    }
private:
    std::span<const Complex> _input;
    std::span<Complex> _output;
};
