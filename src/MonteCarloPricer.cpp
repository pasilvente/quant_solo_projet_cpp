#include "MonteCarloPricer.hpp"
#include <random>
#include <cmath>
#include <algorithm>

MonteCarloPricer::MonteCarloPricer(unsigned int nPaths, unsigned int nSteps)
    : m_nPaths(nPaths), m_nSteps(nSteps) {}

// Pricing Monte Carlo d'une option européenne sous GBM
MonteCarloPricer::Result MonteCarloPricer::price(const Option& option,
                                                 const BlackScholesModel& model) const {
    double S0    = model.spot();
    double r     = model.rate();
    double sigma = model.volatility();
    double K     = option.strike();
    double T     = option.maturity();

    double dt = T / static_cast<double>(m_nSteps);

    // Générateur aléatoire
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    double sum = 0.0;
    double sumSq = 0.0;

    for (unsigned int i = 0; i < m_nPaths; ++i) {
        double S = S0;

        // Simulation de la trajectoire
        for (unsigned int step = 0; step < m_nSteps; ++step) {
            double Z = dist(gen);
            S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        }

        double payoff = 0.0;
        if (option.type() == OptionType::Call) {
            payoff = std::max(S - K, 0.0);
        } else {
            payoff = std::max(K - S, 0.0);
        }

        // Actualisation
        double discPayoff = std::exp(-r * T) * payoff;

        sum   += discPayoff;
        sumSq += discPayoff * discPayoff;
    }

    double n = static_cast<double>(m_nPaths);
    double mean = sum / n;

    double variance = 0.0;
    if (m_nPaths > 1) {
        variance = (sumSq - n * mean * mean) / (n - 1.0); // variance empirique
    }

    double stdError = std::sqrt(variance / n);
    double z = 1.96; // ~95% CI

    Result res;
    res.price   = mean;
    res.stdError = stdError;
    res.ciLower = mean - z * stdError;
    res.ciUpper = mean + z * stdError;

    return res;
}
