#pragma once
#include <complex>
#include "test_utils.h"
#include "../src/fourier_transform.h"

template <typename FTImpl, ft_data_type Vt, Vt TOL>
struct FourierTransformBackend {
    static constexpr Vt tolerance = TOL;
    using value_type = Vt;
    static void compute(const std::span<std::complex<Vt>> input,
                        std::span<std::complex<Vt>> output) {
        FTImpl ft(input, output);
        ft.execute();
    }
};

template <typename FTImpl1, typename FTImpl2, ft_data_type Vt, Vt TOL>
struct FourierTransformComparisonBackend {
    static constexpr Vt tolerance = TOL;
    using value_type = Vt;
    static void compute1(const std::span<std::complex<Vt>> input,
                        std::span<std::complex<Vt>> output) {
        FTImpl1 ft(input, output);
        ft.execute();
    }
    static void compute2(const std::span<std::complex<Vt>> input,
                        std::span<std::complex<Vt>> output) {
        FTImpl2 ft(input, output);
        ft.execute();
    }
};
