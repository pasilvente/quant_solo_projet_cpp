// File: include/pricing/methods/MonteCarloPricer.hpp
#pragma once

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"

/**
 * @brief Paramètres de la simulation Monte Carlo.
 */
struct MCConfig {
    unsigned int nPaths          = 100000;
    unsigned int nSteps          = 100;
    bool         useAntithetic   = false;
    bool         useControlVariate = false;
};

/**
 * @brief Résultat d'un pricing MC : prix + erreur + IC.
 */
struct MCResult {
    double price    = 0.0;
    double stdError = 0.0;
    double ciLower  = 0.0;
    double ciUpper  = 0.0;
};

/**
 * @brief Monte Carlo pricer pour options européennes sous Black–Scholes.
 *
 * Peut utiliser :
 *  - variantes antithétiques,
 *  - une simple control variate basée sur le sous-jacent S_T actualisé.
 */
class MonteCarloPricer {
public:
    explicit MonteCarloPricer(const MCConfig& cfg);

    MCResult price(const Option& option,
                   const BlackScholesModel& model) const;

private:
    MCConfig m_cfg;
};
