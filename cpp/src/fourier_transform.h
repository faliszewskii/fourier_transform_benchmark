#pragma once
#include <complex>

template<typename Vt>
concept ft_data_type = std::same_as<Vt, float> || std::same_as<Vt, double>;

template<typename T, typename Vt>
concept fourier_transform = requires(
    T a,
    const std::span<std::complex<Vt>> &input,
    std::span<std::complex<Vt>> &output
) {
    { T(input, output) };
    { a.execute() } -> std::same_as<void>;
};
