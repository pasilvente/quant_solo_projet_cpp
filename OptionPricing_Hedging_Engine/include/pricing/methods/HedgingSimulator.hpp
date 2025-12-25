// File: include/pricing/methods/HedgingSimulator.hpp
#pragma once

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"

/**
 * @brief Paramètres de la simulation de delta-hedging.
 */
struct HedgingConfig {
    unsigned int nPaths = 10000;  ///< nombre de trajectoires
    unsigned int nSteps = 252;    ///< nombre de rebalancements (ex: 252 jours)
};

/**
 * @brief Statistiques sur le P&L de hedging.
 *
 * On regarde la distribution du P&L final de la position :
 *   short option + delta hedge.
 */
struct HedgingStats {
    double meanPnL = 0.0;
    double stdPnL  = 0.0;
    double minPnL  = 0.0;
    double maxPnL  = 0.0;
};

/**
 * @brief Simulateur de delta-hedging pour option européenne sous Black–Scholes.
 *
 * Hypothèses :
 *  - dynamique sous-jacente : GBM (même que ton pricer MC),
 *  - modèle cohérent avec le pricing BS (même r, sigma),
 *  - hedge avec le Delta analytique Black–Scholes.
 */
class HedgingSimulator {
public:
    explicit HedgingSimulator(const HedgingConfig& cfg);

    /**
     * @brief Simule une stratégie de delta-hedging pour un call/put européen.
     *
     * @param option Option (doit être de style European).
     * @param model  Modèle Black–Scholes (S0, r, sigma).
     *
     * @return Statistiques de P&L (en unité de monnaie).
     */
    HedgingStats simulateDeltaHedge(const Option& option,
                                    const BlackScholesModel& model) const;

private:
    HedgingConfig m_cfg;

    /**
     * @brief Delta analytique Black–Scholes à un temps t et spot S.
     *
     * @param option Option (Call/Put, European).
     * @param model  Paramètres r, sigma (S0 pas utilisé directement ici).
     * @param S      Spot courant.
     * @param t      Temps courant (0 <= t <= T).
     */
    double bsDelta(const Option& option,
                   const BlackScholesModel& model,
                   double S,
                   double t) const;
};
