// File: src/core/Normal.cpp
#include "pricing/core/Normal.hpp"
#include <cmath>

namespace stats {

    double normal_pdf(double x) {
        // 1 / sqrt(2π)
        constexpr double INV_SQRT_2PI = 0.39894228040143267794;
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    double normal_cdf(double x) {
        // Formule standard : N(x) = 0.5 * erfc(-x / sqrt(2))
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

} // namespace stats
