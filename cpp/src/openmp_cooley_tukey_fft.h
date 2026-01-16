#pragma once
#include <cassert>
#include <complex>

template<std::floating_point Vt>
class OpenMpCooleyTukeyFFT {
    using Complex = std::complex<Vt>;
    static constexpr Vt pi = std::numbers::pi_v<Vt>;
public:
    OpenMpCooleyTukeyFFT(std::span<const std::complex<Vt>> input, std::span<std::complex<Vt>> output): _input(input), _output(output) {
        const size_t N = input.size();
        assert((N & N-1) == 0); // Assert that N is power of 2.
    }

    void execute() {
        const size_t N = _input.size();

        int iterations = std::bit_width(N) - 1; // log2(N)
        #pragma omp parallel for schedule(static) default(none) shared(_input, _output, N, iterations)
        for (int i = 0; i < N; ++i) { // Shuffle input
            _output[i] = _input[mirrorBits(i, iterations)];
        }

        for (int it = 0; it < iterations; ++it) {
            const int groupCount = 1 << (iterations - it - 1);
            const int groupSize = N / groupCount;
            const int jump = groupSize / 2;

            #pragma omp parallel for schedule(static) default(none) shared(_output, N, groupCount, groupSize, jump)
            for (int n = 0; n < N/2; ++n) {
                int group = n / (groupSize/2);
                int node = n % (groupSize/2);
                auto angle = -2 * pi * node / groupSize;
                auto w =  Complex(cos(angle), sin(angle));
                int k = group * groupSize + node;
                auto u = _output[k];
                auto t = w * _output[k + jump];
                _output[k] = u + t;
                _output[k + jump] = u - t;
            }
        }
    }
private:
    static int mirrorBits(int n, int bits) {
        int r = 0;
        for (int i = 0; i < bits; i++) {
            r <<= 1;
            r |= n & 1;
            n >>= 1;
        }
        return r;
    }

    std::span<const std::complex<Vt>> _input;
    std::span<std::complex<Vt>> _output;
};