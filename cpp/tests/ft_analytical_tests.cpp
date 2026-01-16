#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>

#include "ft_backends.h"
#include "../src/fftw_wrapper.h"
#include "../src/naive_dft.h"
#include "../src/cooley_tukey_fft.h"
#include "../src/openmp_cooley_tukey_fft.h"

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
        ExpectComplexNear(output[k], {0.0, 0.0});
}

TYPED_TEST_P(FourierTransformAnalyticalTest, UnitImpulse) {
    const int N = 16;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(0.0, 0.0));
    input[0] = 1.0;

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    for (int k = 0; k < N; ++k)
        ExpectComplexNear(output[k], {1.0, 0.0});
}

TYPED_TEST_P(FourierTransformAnalyticalTest, ConstantSignal) {
    const int N = 4;
    const double C = 2.5;

    using Complex = std::complex<typename TypeParam::value_type>;
    std::vector input(N, Complex(C, 0.0));

    std::vector output(N, Complex(0.0, 0.0));
    TypeParam::compute(input, output);

    ExpectComplexNear(output[0], {N * C, 0.0});

    for (int k = 1; k < N; ++k)
        ExpectComplexNear(output[k], {0.0, 0.0});
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

    ExpectComplexNear(output[k],     {N / 2.0, 0.0});
    ExpectComplexNear(output[N - k], {N / 2.0, 0.0});

    for (int i = 0; i < N; ++i) {
        if (i != k && i != N - k)
            EXPECT_NEAR(std::abs(output[i]), 0.0, TOL);
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

    ExpectComplexNear(output[k],     {0.0, -N / 2.0});
    ExpectComplexNear(output[N - k], {0.0,  N / 2.0});

    for (int i = 0; i < N; ++i) {
        if (i != k && i != N - k)
            EXPECT_NEAR(std::abs(output[i]), 0.0, TOL);
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

    ExpectComplexNear(output[N / 2], {N, 0.0});

    for (int k = 0; k < N; ++k) {
        if (k != N / 2)
            ExpectComplexNear(output[k], {0.0, 0.0});
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
        EXPECT_NEAR(output[k].imag(), 0.0, TOL);
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
        EXPECT_NEAR(output[k].real(), 0.0, TOL);
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

    EXPECT_NEAR(output[0].real(), 8, TOL);
    EXPECT_NEAR(output[1].real(), 0, TOL);
    EXPECT_NEAR(output[2].real(), 4, TOL);
    EXPECT_NEAR(output[3].real(), 0, TOL);
    EXPECT_NEAR(output[4].real(), 0, TOL);
    EXPECT_NEAR(output[5].real(), 0, TOL);
    EXPECT_NEAR(output[6].real(), 4, TOL);
    EXPECT_NEAR(output[7].real(), 0, TOL);
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

using FFTImplementations = ::testing::Types<
    FourierTransformBackend<FftwWrapper, double>,
    FourierTransformBackend<NaiveDFT<double>, double>,
    FourierTransformBackend<CooleyTukeyFFT<double>, double>,
    FourierTransformBackend<OpenMpCooleyTukeyFFT<double>, double>
>;


INSTANTIATE_TYPED_TEST_SUITE_P(
    FourierTransformBackends,
    FourierTransformAnalyticalTest,
    FFTImplementations
);