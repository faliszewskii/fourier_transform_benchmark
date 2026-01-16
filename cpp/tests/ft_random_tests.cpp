#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <random>

#include "ft_backends.h"
#include "../src/fftw_wrapper.h"
#include "../src/naive_dft.h"
#include "../src/cooley_tukey_fft.h"
#include "../src/openmp_cooley_tukey_fft.h"

template <typename Backend>
class FourierTransformRandomTest : public testing::Test {};

TYPED_TEST_SUITE_P(FourierTransformRandomTest);

TYPED_TEST_P(FourierTransformRandomTest, RandomTest) {
    constexpr size_t N = 1 << 10;

    using ValueType = typename TypeParam::value_type;
    using Complex = std::complex<ValueType>;
    std::vector input(N, Complex(0.0, 0.0));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<ValueType> dist(0.0, 1.0);

    for (auto& v : input) {
        v = Complex(dist(rng), dist(rng));
    }

    std::vector output1(N, std::complex(0.0, 0.0));
    TypeParam::compute1(input, output1);

    std::vector output2(N, std::complex(0.0, 0.0));
    TypeParam::compute2(input, output2);

    for (int k = 0; k < N; ++k)
        ExpectComplexNear(output1[k], output2[k]);
}

REGISTER_TYPED_TEST_SUITE_P(
    FourierTransformRandomTest,
    RandomTest
);

using FFTImplementations = ::testing::Types<
    FourierTransformComparisonBackend<NaiveDFT<double>, FftwWrapper, double>,
    FourierTransformComparisonBackend<CooleyTukeyFFT<double>, FftwWrapper, double>,
    FourierTransformComparisonBackend<OpenMpCooleyTukeyFFT<double>, FftwWrapper, double>
>;


INSTANTIATE_TYPED_TEST_SUITE_P(
    FourierTransformBackends,
    FourierTransformRandomTest,
    FFTImplementations
);