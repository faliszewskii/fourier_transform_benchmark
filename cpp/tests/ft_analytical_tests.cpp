#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>

#include "ft_backends.h"
#include "../src/fftw_wrapper.h"
#include "../src/naive_dft.h"
#include "../src/cooley_tukey_fft.h"
#include "../src/cuda/cu_fft.cuh"
#include "../src/openmp_cooley_tukey_fft.h"
#include "../src/cuda/cuda_cooley_tukey.h"

template <typename Backend>
class FourierTransformAnalyticalTest : public testing::Test {};

TYPED_TEST_SUITE_P(FourierTransformAnalyticalTest);

TYPED_TEST_P(FourierTransformAnalyticalTest, ZeroSignal) {
    const int N = 16;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    for (int k = 0; k < N; ++k)
        ExpectComplexNear(output[k], {0.0, 0.0}, TypeParam::tolerance);
}

TYPED_TEST_P(FourierTransformAnalyticalTest, UnitImpulse) {
    const int N = 16;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));
    input[0] = 1.0;

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    for (int k = 0; k < N; ++k)
        ExpectComplexNear(output[k], {1.0, 0.0}, TypeParam::tolerance);
}

TYPED_TEST_P(FourierTransformAnalyticalTest, ConstantSignal) {
    const int N = 4;
    const double C = 2.5;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(C, 0.0));

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    ExpectComplexNear(output[0], {N * C, 0.0}, TypeParam::tolerance);

    for (int k = 1; k < N; ++k)
        ExpectComplexNear(output[k], {0.0, 0.0}, TypeParam::tolerance);
}

TYPED_TEST_P(FourierTransformAnalyticalTest, BinAlignedCosine) {
    const int N = 64;
    const int k = 7;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));
    for (int n = 0; n < N; ++n)
        input[n] = std::cos(2.0 * M_PI * k * n / N);

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    ExpectComplexNear(output[k],     {N / 2.0, 0.0}, TypeParam::tolerance);
    ExpectComplexNear(output[N - k], {N / 2.0, 0.0}, TypeParam::tolerance);

    for (int i = 0; i < N; ++i) {
        if (i != k && i != N - k)
            EXPECT_NEAR(std::abs(output[i]), 0.0, TypeParam::tolerance);
    }
}

TYPED_TEST_P(FourierTransformAnalyticalTest, BinAlignedSine) {
    const int N = 64;
    const int k = 5;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));
    for (int n = 0; n < N; ++n)
        input[n] = std::sin(2.0 * M_PI * k * n / N);

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    ExpectComplexNear(output[k],     {0.0, -N / 2.0}, TypeParam::tolerance);
    ExpectComplexNear(output[N - k], {0.0,  N / 2.0}, TypeParam::tolerance);

    for (int i = 0; i < N; ++i) {
        if (i != k && i != N - k)
            EXPECT_NEAR(std::abs(output[i]), 0.0, TypeParam::tolerance);
    }
}

TYPED_TEST_P(FourierTransformAnalyticalTest, NyquistFrequency) {
    const int N = 32;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));

    for (int n = 0; n < N; ++n)
        input[n] = (n % 2 == 0) ? 1.0 : -1.0;

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    ExpectComplexNear(output[N / 2], {N, 0.0}, TypeParam::tolerance);

    for (int k = 0; k < N; ++k) {
        if (k != N / 2)
            ExpectComplexNear(output[k], {0.0, 0.0}, TypeParam::tolerance);
    }
}

TYPED_TEST_P(FourierTransformAnalyticalTest, CircularEvenSymmetryRealSpectrum) {
    const int N = 64;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));

    input[0] = 1.0;
    input[N / 2] = 0.5;

    for (int k = 1; k < N / 2; ++k) {
        double v = std::cos(0.2 * k);
        input[k] = v;
        input[N - k] = v;
    }

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    for (int k = 0; k < N; ++k)
        EXPECT_NEAR(output[k].imag(), 0.0, TypeParam::tolerance);
}


TYPED_TEST_P(FourierTransformAnalyticalTest, OddSymmetryImagSpectrum) {
    const int N = 64;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));

    for (int n = 1; n < N / 2; ++n) {
        input[n] =  std::sin(0.3 * n);
        input[N - n] = -input[n];
    }

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    for (int k = 0; k < N; ++k)
        EXPECT_NEAR(output[k].real(), 0.0, TypeParam::tolerance);
}

TYPED_TEST_P(FourierTransformAnalyticalTest, SimpleSignal) {
    const int N = 8;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));
    input[0] = 2;
    input[1] = 1;
    input[2] = 0;
    input[3] = 1;
    input[4] = 2;
    input[5] = 1;
    input[6] = 0;
    input[7] = 1;

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    EXPECT_NEAR(output[0].real(), 8, TypeParam::tolerance);
    EXPECT_NEAR(output[1].real(), 0, TypeParam::tolerance);
    EXPECT_NEAR(output[2].real(), 4, TypeParam::tolerance);
    EXPECT_NEAR(output[3].real(), 0, TypeParam::tolerance);
    EXPECT_NEAR(output[4].real(), 0, TypeParam::tolerance);
    EXPECT_NEAR(output[5].real(), 0, TypeParam::tolerance);
    EXPECT_NEAR(output[6].real(), 4, TypeParam::tolerance);
    EXPECT_NEAR(output[7].real(), 0, TypeParam::tolerance);
}

REGISTER_TYPED_TEST_SUITE_P(
    FourierTransformAnalyticalTest,
    ZeroSignal,
    UnitImpulse,
    ConstantSignal,
    BinAlignedCosine,
    BinAlignedSine,
    NyquistFrequency,
    CircularEvenSymmetryRealSpectrum,
    OddSymmetryImagSpectrum,
    SimpleSignal
);

static constexpr float FLOAT_TOL = std::numeric_limits<float>::epsilon() * 10000;
static constexpr double DOUBLE_TOL = std::numeric_limits<double>::epsilon() * 10000;
using FFTImplementations = ::testing::Types<
    FourierTransformBackend<FftwWrapper<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<NaiveDFT<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<CooleyTukeyFFT<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<OpenMpCooleyTukeyFFT<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<CuFFTWrapper<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<CudaCooleyTukeyFFT<double>, double, DOUBLE_TOL>,
    FourierTransformBackend<FftwWrapper<float>, float, FLOAT_TOL>,
    FourierTransformBackend<NaiveDFT<float>, float, FLOAT_TOL>,
    FourierTransformBackend<CooleyTukeyFFT<float>, float, FLOAT_TOL>,
    FourierTransformBackend<OpenMpCooleyTukeyFFT<float>, float, FLOAT_TOL>,
    FourierTransformBackend<CuFFTWrapper<float>, float, FLOAT_TOL>,
    FourierTransformBackend<CudaCooleyTukeyFFT<float>, float, FLOAT_TOL>
>;


INSTANTIATE_TYPED_TEST_SUITE_P(
    FourierTransformBackends,
    FourierTransformAnalyticalTest,
    FFTImplementations
);