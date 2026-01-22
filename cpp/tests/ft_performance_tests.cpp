#include <benchmark/benchmark.h>

#include <vector>
#include <complex>
#include <random>

#include "../src/fftw_wrapper.h"
#include "../src/naive_dft.h"
#include "../src/cooley_tukey_fft.h"
#include "../src/cu_fft.cuh"
#include "../src/fourier_transform.h"
#include "../src/openmp_cooley_tukey_fft.h"


template <ft_data_type Vt, fourier_transform<Vt> FourierTransform>
static void BM_FourierTransform_BigExample(benchmark::State& state)
{
    const size_t N = state.range(0);

    std::vector<std::complex<Vt>> input(N);
    std::vector<std::complex<Vt>> output(N);
    FourierTransform fft(input, output);

    std::mt19937 rng(123456);
    std::uniform_real_distribution<Vt> dist(0.0, 1.0);

    for (auto& v : input) {
        v = std::complex<Vt>(dist(rng), dist(rng));
    }

    for (auto _ : state) {
        fft.execute();
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * N);
    state.SetBytesProcessed(
        state.iterations() * N * sizeof(std::complex<Vt>)
    );
}

BENCHMARK_TEMPLATE(BM_FourierTransform_BigExample, float, FftwWrapper<float>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 26)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_FourierTransform_BigExample, float, NaiveDFT<float>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 14)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_FourierTransform_BigExample, float, CooleyTukeyFFT<float>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 26)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_FourierTransform_BigExample, float, OpenMpCooleyTukeyFFT<float>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 26)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_FourierTransform_BigExample, float, CuFFTWrapper<float>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 26)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
