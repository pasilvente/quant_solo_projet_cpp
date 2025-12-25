// File: src/methods/MonteCarloPricer.cpp
#include "pricing/methods/MonteCarloPricer.hpp"

#include <random>
#include <cmath>
#include <algorithm>

/*
 * Idée :
 *  - On simule des trajectoires GBM.
 *  - On calcule le payoff actualisé Y_i = e^{-rT} payoff(S_T).
 *  - Variable de contrôle : CV_i = e^{-rT} S_T, avec E[CV] = S0.
 *  - Si useControlVariate : on corrige le mean via
 *        Y*_i = Y_i - α (CV_i - E[CV]), α = Cov(Y,CV) / Var(CV).
 */

MonteCarloPricer::MonteCarloPricer(const MCConfig& cfg)
    : m_cfg(cfg) {}

MCResult MonteCarloPricer::price(const Option& option,
                                 const BlackScholesModel& model) const {
    const double S0    = model.spot();
    const double r     = model.rate();
    const double sigma = model.volatility();
    const double K     = option.strike();
    const double T     = option.maturity();

    const double dt  = T / static_cast<double>(m_cfg.nSteps);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double volStep = sigma * std::sqrt(dt);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    // Accumulateurs pour statistiques
    double sumY   = 0.0;
    double sumY2  = 0.0;
    double sumCV  = 0.0;
    double sumCV2 = 0.0;
    double sumYCV = 0.0;

    std::size_t nEffSamples = 0; // nb effectif d'échantillons (double si antithetic)

    auto processSample = [&](double ST) {
        double payoff = 0.0;
        if (option.type() == OptionType::Call) {
            payoff = std::max(ST - K, 0.0);
        } else {
            payoff = std::max(K - ST, 0.0);
        }

        const double disc = std::exp(-r * T);
        const double Y  = disc * payoff;  // ce qu'on veut estimer
        const double CV = disc * ST;      // variable de contrôle (E[CV] = S0)

        sumY   += Y;
        sumY2  += Y * Y;
        sumCV  += CV;
        sumCV2 += CV * CV;
        sumYCV += Y * CV;

        ++nEffSamples;
    };

    for (unsigned int path = 0; path < m_cfg.nPaths; ++path) {
        double S_plus = S0;
        double S_minus = S0;

        if (m_cfg.useAntithetic) {
            for (unsigned int step = 0; step < m_cfg.nSteps; ++step) {
                const double Z = dist(gen);
                const double incrPlus  = std::exp(drift + volStep * Z);
                const double incrMinus = std::exp(drift - volStep * Z);
                S_plus  *= incrPlus;
                S_minus *= incrMinus;
            }
            processSample(S_plus);
            processSample(S_minus);
        } else {
            for (unsigned int step = 0; step < m_cfg.nSteps; ++step) {
                const double Z = dist(gen);
                const double incr = std::exp(drift + volStep * Z);
                S_plus *= incr;
            }
            processSample(S_plus);
        }
    }

    const double n = static_cast<double>(nEffSamples);
    MCResult res{};

    if (n < 2.0) {
        // Cas pathologique, on renvoie juste 0.
        return res;
    }

    const double meanY  = sumY / n;
    const double meanCV = sumCV / n;

    const double varY = (sumY2 - n * meanY * meanY) / (n - 1.0);
    const double varCV = (sumCV2 - n * meanCV * meanCV) / (n - 1.0);
    const double covYCV = (sumYCV - n * meanY * meanCV) / (n - 1.0);

    double estMean = meanY;
    double estVar  = varY;

    if (m_cfg.useControlVariate && varCV > 1e-14) {
        const double expectedCV = S0; // E[e^{-rT} S_T] = S0 sous risque neutre
        const double alpha = covYCV / varCV;

        // Nouvelle moyenne
        estMean = meanY - alpha * (meanCV - expectedCV);
        // Nouvelle variance : Var(Y - α CV)
        estVar  = varY + alpha * alpha * varCV - 2.0 * alpha * covYCV;
    }

    const double stdErr = std::sqrt(estVar / n);
    const double z = 1.96;

    res.price    = estMean;
    res.stdError = stdErr;
    res.ciLower  = estMean - z * stdErr;
    res.ciUpper  = estMean + z * stdErr;

    return res;
}
