#pragma once

#include <complex>
#include <gtest/gtest.h>
#include <fftw3.h>

inline void ExpectComplexNear(const std::complex<double>& a,
                              const std::complex<double>& b,
                              float tol) {
    EXPECT_NEAR(a.real(), b.real(), tol);
    EXPECT_NEAR(a.imag(), b.imag(), tol);
}
