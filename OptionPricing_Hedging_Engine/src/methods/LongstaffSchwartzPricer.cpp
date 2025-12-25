// File: src/methods/LongstaffSchwartzPricer.cpp
#include "pricing/methods/LongstaffSchwartzPricer.hpp"

#include <random>
#include <cmath>
#include <numeric>
#include <stdexcept>

/*
 * Remarques :
 *  - On travaille avec m_nSteps dates d'exercice discrètes (incluant maturité).
 *  - On garde, pour chaque trajectoire i :
 *      - tau[i]        : indice de la date d'exercice choisie,
 *      - payoffTau[i]  : payoff à cette date.
 *  - A chaque pas j, on régresse la valeur de continuation (payoffTau discounted
 *    de tau[i] vers j) sur {1, S_j, S_j^2} à partir des trajectoires ITM
 *    qui ne sont pas encore exercées.
 */

LongstaffSchwartzPricer::LongstaffSchwartzPricer(unsigned int nPaths,
                                                 unsigned int nSteps)
    : m_nPaths(nPaths),
      m_nSteps(nSteps) {
    if (nPaths < 1000 || nSteps < 2) {
        throw std::invalid_argument("LSMC: use at least ~1000 paths and >= 2 steps");
    }
}

LongstaffSchwartzPricer::RegressionCoeffs
LongstaffSchwartzPricer::regressContinuation(const std::vector<double>& S,
                                             const std::vector<double>& Y) const {
    const std::size_t n = S.size();
    RegressionCoeffs coeffs{0.0, 0.0, 0.0};

    if (n < 3) {
        // Trop peu de points : on prend juste une constante = mean(Y).
        double sumY = std::accumulate(Y.begin(), Y.end(), 0.0);
        coeffs.a0 = sumY / static_cast<double>(n);
        return coeffs;
    }

    double sum1   = static_cast<double>(n);
    double sumX   = 0.0;
    double sumX2  = 0.0;
    double sumX3  = 0.0;
    double sumX4  = 0.0;
    double sumY   = 0.0;
    double sumXY  = 0.0;
    double sumX2Y = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double x = S[i];
        const double y = Y[i];
        const double x2 = x * x;

        sumX   += x;
        sumX2  += x2;
        sumX3  += x2 * x;
        sumX4  += x2 * x2;
        sumY   += y;
        sumXY  += x * y;
        sumX2Y += x2 * y;
    }

    // Normal equations pour polynôme de degré 2.
    double A[3][3] = {
        { sum1,  sumX,  sumX2 },
        { sumX,  sumX2, sumX3 },
        { sumX2, sumX3, sumX4 }
    };
    double b[3] = { sumY, sumXY, sumX2Y };

    // Gauss simple 3x3
    for (int i = 0; i < 3; ++i) {
        // pivot
        double pivot = A[i][i];
        if (std::fabs(pivot) < 1e-12) {
            // Matrice mal conditionnée : fallback à constante
            coeffs.a0 = sumY / sum1;
            coeffs.a1 = 0.0;
            coeffs.a2 = 0.0;
            return coeffs;
        }

        for (int j = i; j < 3; ++j) {
            A[i][j] /= pivot;
        }
        b[i] /= pivot;

        for (int k = i + 1; k < 3; ++k) {
            double factor = A[k][i];
            for (int j = i; j < 3; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back-substitution
    double x2 = b[2];
    double x1 = b[1] - A[1][2] * x2;
    double x0 = b[0] - A[0][1] * x1 - A[0][2] * x2;

    coeffs.a0 = x0;
    coeffs.a1 = x1;
    coeffs.a2 = x2;

    return coeffs;
}

LSMCResult LongstaffSchwartzPricer::price(const Option& option,
                                          const BlackScholesModel& model,
                                          const Payoff& payoff) const {
    if (option.style() != ExerciseStyle::American) {
        throw std::invalid_argument("LSMC pricer expects an American option");
    }

    const double S0    = model.spot();
    const double r     = model.rate();
    const double sigma = model.volatility();
    const double T     = option.maturity();

    const double dt    = T / static_cast<double>(m_nSteps);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double volStep = sigma * std::sqrt(dt);

    // Simulation des trajectoires S_{t_j}^i
    std::vector<std::vector<double>> paths(m_nPaths,
                                           std::vector<double>(m_nSteps + 1));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    for (unsigned int i = 0; i < m_nPaths; ++i) {
        double S = S0;
        paths[i][0] = S;
        for (unsigned int j = 1; j <= m_nSteps; ++j) {
            const double Z = dist(gen);
            S *= std::exp(drift + volStep * Z);
            paths[i][j] = S;
        }
    }

    // tau[i] : indice de la date d'exercice retenue (0..m_nSteps)
    std::vector<unsigned int> tau(m_nPaths, m_nSteps);
    std::vector<double> payoffTau(m_nPaths);

    // Initialisation : à maturité, exercice possible
    for (unsigned int i = 0; i < m_nPaths; ++i) {
        const double ST = paths[i][m_nSteps];
        payoffTau[i] = payoff(ST); // 0 si OTM
    }

    // Backward induction : j = m_nSteps-1 .. 1
    for (int j = static_cast<int>(m_nSteps) - 1; j >= 1; --j) {
        std::vector<double> S_itm;
        std::vector<double> Y_itm;

        // Construction des données de régression (trajectoires ITM et non exercées)
        for (unsigned int i = 0; i < m_nPaths; ++i) {
            if (tau[i] <= static_cast<unsigned int>(j))
                continue; // déjà exercée avant ou à j

            const double Sj = paths[i][j];
            const double hj = payoff(Sj);

            if (hj <= 1e-14)
                continue; // out-of-the-money

            // Valeur future si on n'exerce pas à j : payoffTau[i] à tau[i].
            const double t_diff = (tau[i] - j) * dt;
            const double contValDiscounted = payoffTau[i] * std::exp(-r * t_diff);

            S_itm.push_back(Sj);
            Y_itm.push_back(contValDiscounted);
        }

        if (S_itm.size() < 3) {
            // Trop peu de points pour une régression fiable : on ne change rien à ce pas.
            continue;
        }

        // Régression de la continuation value
        RegressionCoeffs coeffs = regressContinuation(S_itm, Y_itm);

        // Décision exercice vs continuation
        for (unsigned int i = 0; i < m_nPaths; ++i) {
            if (tau[i] <= static_cast<unsigned int>(j))
                continue;

            const double Sj = paths[i][j];
            const double hj = payoff(Sj);

            if (hj <= 1e-14)
                continue;

            const double cont = coeffs.a0 + coeffs.a1 * Sj + coeffs.a2 * Sj * Sj;

            if (hj >= cont) {
                // Exercice optimal à t_j sur cette trajectoire
                tau[i] = static_cast<unsigned int>(j);
                payoffTau[i] = hj;
            }
        }
    }

    // Prix = moyenne des payoffs actualisés à t=0
    std::vector<double> discounted(m_nPaths);
    for (unsigned int i = 0; i < m_nPaths; ++i) {
        const double t_i = tau[i] * dt;
        discounted[i] = payoffTau[i] * std::exp(-r * t_i);
    }

    double sum = 0.0;
    double sum2 = 0.0;
    for (double v : discounted) {
        sum  += v;
        sum2 += v * v;
    }

    const double n = static_cast<double>(m_nPaths);
    const double mean = sum / n;
    const double var  = (sum2 - n * mean * mean) / (n - 1.0);
    const double stdErr = std::sqrt(var / n);

    LSMCResult res;
    res.price    = mean;
    res.stdError = stdErr;
    return res;
}
