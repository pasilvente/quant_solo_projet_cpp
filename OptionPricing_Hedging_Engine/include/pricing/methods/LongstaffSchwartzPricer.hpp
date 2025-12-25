// File: include/pricing/methods/LongstaffSchwartzPricer.hpp
#pragma once

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"
#include "pricing/core/Payoff.hpp"
#include <vector>

/**
 * @brief Résultat du pricing LSMC : prix + erreur standard.
 */
struct LSMCResult {
    double price    = 0.0;
    double stdError = 0.0;
};

/**
 * @brief Longstaff–Schwartz Monte Carlo pour options américaines.
 *
 * Implémente l'algorithme LSMC classique :
 *  - simulation de trajectoires GBM,
 *  - travail à rebours,
 *  - régression polynomiale (1, S, S²) pour approximer la continuation value.
 */
class LongstaffSchwartzPricer {
public:
    LongstaffSchwartzPricer(unsigned int nPaths,
                            unsigned int nSteps);

    LSMCResult price(const Option& option,
                     const BlackScholesModel& model,
                     const Payoff& payoff) const;

private:
    unsigned int m_nPaths;
    unsigned int m_nSteps;

    struct RegressionCoeffs {
        double a0;
        double a1;
        double a2;
    };

    RegressionCoeffs regressContinuation(const std::vector<double>& S,
                                         const std::vector<double>& Y) const;
};
