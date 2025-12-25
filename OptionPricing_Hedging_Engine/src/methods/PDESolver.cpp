// File: src/methods/PDESolver.cpp
#include "pricing/methods/PDESolver.hpp"
#include "pricing/util/LinearSolver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

PDESolver::PDESolver(double T,
                     double r,
                     double sigma,
                     double S_max,
                     std::size_t N,
                     std::size_t M)
    : m_T(T),
      m_r(r),
      m_sigma(sigma),
      m_S_max(S_max),
      m_N(N),
      m_M(M),
      m_dt(T / static_cast<double>(M)),
      m_dS(S_max / static_cast<double>(N)),
      m_B_lower(N - 1),
      m_B_diag(N - 1),
      m_B_upper(N - 1),
      m_A_lower(N - 1),
      m_A_diag(N - 1),
      m_A_upper(N - 1) {
    if (N < 3 || M < 1) {
        throw std::invalid_argument("PDESolver: N must be >= 3 and M >= 1");
    }
}

void PDESolver::precomputeMatrices() {
    const double sigma2 = m_sigma * m_sigma;

    // Points intérieurs i = 1..N-1 (les bords 0 et N sont traités à part).
    for (std::size_t i = 1; i < m_N; ++i) {
        const std::size_t j = i - 1; // index 0..N-2 dans les vecteurs
        const double S_i = i * m_dS;

        const double alpha = 0.5 * sigma2 * S_i * S_i / (m_dS * m_dS)
                           - 0.5 * m_r * S_i / m_dS;
        const double beta  = - sigma2 * S_i * S_i / (m_dS * m_dS)
                             - m_r;
        const double gamma = 0.5 * sigma2 * S_i * S_i / (m_dS * m_dS)
                           + 0.5 * m_r * S_i / m_dS;

        // Matrice implicite A : (I - 0.5 dt L)
        m_A_lower[j] = -0.5 * m_dt * alpha;
        m_A_diag[j]  = 1.0 - 0.5 * m_dt * beta;
        m_A_upper[j] = -0.5 * m_dt * gamma;

        // Matrice explicite B : (I + 0.5 dt L)
        m_B_lower[j] = 0.5 * m_dt * alpha;
        m_B_diag[j]  = 1.0 + 0.5 * m_dt * beta;
        m_B_upper[j] = 0.5 * m_dt * gamma;
    }
}

PricingResults PDESolver::solve(const Payoff& payoff,
                                double S0,
                                bool isAmerican) {
    // Grille en S : S_i = i * dS, i = 0..N
    std::vector<double> S(m_N + 1);
    for (std::size_t i = 0; i <= m_N; ++i) {
        S[i] = i * m_dS;
    }

    // Condition terminale t = T : V^M_i = payoff(S_i)
    std::vector<double> V_old(m_N + 1), V_new(m_N + 1);
    for (std::size_t i = 0; i <= m_N; ++i) {
        V_old[i] = payoff(S[i]);
    }

    // Remontée en temps de t = T vers t = 0
    for (std::size_t n = 0; n < m_M; ++n) {
        const double t = m_T - (n + 1) * m_dt; // temps après un pas

        // Conditions aux bords.
        // Européenne : on actualise le payoff aux bornes.
        // Américain : la condition d'obstacle sera appliquée ensuite.
        const double V_left  = payoff(0.0)      * std::exp(-m_r * (m_T - t));
        const double V_right = payoff(m_S_max)  * std::exp(-m_r * (m_T - t));

        V_new[0]   = V_left;
        V_new[m_N] = V_right;

        // Système tridiagonal pour les points intérieurs 1..N-1
        const std::size_t nInterior = m_N - 1;
        std::vector<double> a(nInterior), b(nInterior), c(nInterior), d(nInterior);

        for (std::size_t i = 1; i < m_N; ++i) {
            const std::size_t j = i - 1;

            // Terme de droite : B * V_old
            double rhs = m_B_diag[j] * V_old[i];

            if (i > 1) {
                rhs += m_B_lower[j] * V_old[i - 1];
            }
            if (i < m_N - 1) {
                rhs += m_B_upper[j] * V_old[i + 1];
            }

            // Contributions des conditions aux bords via A
            if (i == 1) {
                rhs += m_A_lower[j] * V_left;
            }
            if (i == m_N - 1) {
                rhs += m_A_upper[j] * V_right;
            }

            d[j] = rhs;

            // Coefficients de la matrice A
            a[j] = (i == 1)       ? 0.0 : m_A_lower[j];
            b[j] = m_A_diag[j];
            c[j] = (i == m_N - 1) ? 0.0 : m_A_upper[j];
        }

        // Résolution du système tridiagonal A x = d
        std::vector<double> x;
        Solver::thomasAlgorithm(a, b, c, d, x);

        // Reconstruction de V_new pour les points intérieurs
        for (std::size_t i = 1; i < m_N; ++i) {
            V_new[i] = x[i - 1];
        }

        // Si option américaine : on applique l'obstacle V >= payoff(S)
        if (isAmerican) {
            for (std::size_t i = 0; i <= m_N; ++i) {
                const double intrinsic = payoff(S[i]);
                V_new[i] = std::max(V_new[i], intrinsic);
            }
        }

        V_old.swap(V_new);
    }

    // À ce stade, V_old approxime V(S, t=0).

    PricingResults res;

    // Recherche de l'intervalle S[i0] <= S0 <= S[i0+1]
    std::size_t i0 = 0;
    while (i0 < m_N && S[i0 + 1] < S0) {
        ++i0;
    }
    if (i0 >= m_N) {
        i0 = m_N - 1;
    }

    const double S_low  = S[i0];
    const double S_high = S[i0 + 1];
    const double V_low  = V_old[i0];
    const double V_high = V_old[i0 + 1];

    const double w = (S0 - S_low) / (S_high - S_low);
    res.price = (1.0 - w) * V_low + w * V_high;

    // Approximation de Delta et Gamma par différences finies centrées
    if (i0 > 0 && i0 + 1 < m_N) {
        const double V_im1 = V_old[i0 - 1];
        const double V_i   = V_old[i0];
        const double V_ip1 = V_old[i0 + 1];
        const double dS    = m_dS;

        res.delta = (V_ip1 - V_im1) / (2.0 * dS);
        res.gamma = (V_ip1 - 2.0 * V_i + V_im1) / (dS * dS);
    } else {
        res.delta = 0.0;
        res.gamma = 0.0;
    }

    // Theta laissé à 0 ici (on pourrait l'estimer en regardant la variation en temps).
    res.theta = 0.0;

    return res;
}
