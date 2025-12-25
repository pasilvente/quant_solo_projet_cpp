// File: src/util/LinearSolver.cpp
#include "pricing/util/LinearSolver.hpp"

namespace Solver {

    void thomasAlgorithm(const std::vector<double>& a,
                         const std::vector<double>& b,
                         const std::vector<double>& c,
                         const std::vector<double>& d,
                         std::vector<double>& x) {
        const std::size_t n = b.size();

        if (a.size() != n || c.size() != n || d.size() != n) {
            throw std::invalid_argument(
                "thomasAlgorithm: vector sizes are inconsistent"
            );
        }

        if (n == 0) {
            x.clear();
            return;
        }

        std::vector<double> c_prime(n);
        std::vector<double> d_prime(n);

        // Forward sweep
        double denom = b[0];
        if (std::abs(denom) < 1e-14) {
            throw std::runtime_error("thomasAlgorithm: singular system (denom=0 at i=0)");
        }

        c_prime[0] = (n > 1) ? c[0] / denom : 0.0;
        d_prime[0] = d[0] / denom;

        for (std::size_t i = 1; i < n; ++i) {
            denom = b[i] - a[i] * c_prime[i - 1];
            if (std::abs(denom) < 1e-14) {
                throw std::runtime_error("thomasAlgorithm: singular system (denom=0)");
            }

            c_prime[i] = (i == n - 1) ? 0.0 : c[i] / denom;
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
        }

        // Back substitution
        x.resize(n);
        x[n - 1] = d_prime[n - 1];

        for (std::size_t i = n - 1; i-- > 0;) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
    }

} // namespace Solver
