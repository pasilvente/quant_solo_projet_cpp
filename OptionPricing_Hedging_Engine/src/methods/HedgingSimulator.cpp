// File: src/methods/HedgingSimulator.cpp
#include "pricing/methods/HedgingSimulator.hpp"
#include "pricing/core/Normal.hpp"

#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

HedgingSimulator::HedgingSimulator(const HedgingConfig& cfg)
    : m_cfg(cfg) {
    if (cfg.nPaths < 1000 || cfg.nSteps < 2) {
        throw std::invalid_argument(
            "HedgingSimulator: use at least ~1000 paths and >= 2 steps"
        );
    }
}

double HedgingSimulator::bsDelta(const Option& option,
                                 const BlackScholesModel& model,
                                 double S,
                                 double t) const {
    const double K     = option.strike();
    const double T     = option.maturity();
    const double r     = model.rate();
    const double sigma = model.volatility();

    const double tau = T - t; // temps restant
    if (tau <= 0.0) {
        // À maturité, delta = 0 ou 1 (call) / 0 ou -1 (put) selon ITM/OTM.
        if (option.type() == OptionType::Call) {
            return (S > K) ? 1.0 : 0.0;
        } else {
            return (S < K) ? -1.0 : 0.0;
        }
    }

    const double sqrtTau = std::sqrt(tau);
    const double sigma2  = sigma * sigma;
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma2) * tau)
                      / (sigma * sqrtTau);

    using stats::normal_cdf;

    if (option.type() == OptionType::Call) {
        return normal_cdf(d1);          // delta call
    } else {
        return normal_cdf(d1) - 1.0;    // delta put
    }
}

HedgingStats HedgingSimulator::simulateDeltaHedge(const Option& option,
                                                  const BlackScholesModel& model) const {
    if (option.style() != ExerciseStyle::European) {
        throw std::invalid_argument(
            "HedgingSimulator: only European options are supported for now"
        );
    }

    const double S0    = model.spot();
    const double r     = model.rate();
    const double sigma = model.volatility();
    const double K     = option.strike();
    const double T     = option.maturity();

    const double dt    = T / static_cast<double>(m_cfg.nSteps);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double volStep = sigma * std::sqrt(dt);

    // RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    // On va stocker les P&L finaux pour toutes les trajectoires
    std::vector<double> pnls(m_cfg.nPaths);

    for (unsigned int path = 0; path < m_cfg.nPaths; ++path) {
        double S = S0;
        double t = 0.0;

        // 1) On vend l'option au prix "juste" (théorique).
        //    Pour le pricer analytique, on pourrait appeler ton AnalyticalPricer,
        //    mais ici on simplifie en notant juste que le prix initial C0 sert
        //    de point de départ du compte cash.
        //
        //    Stratégie :
        //      - short 1 option -> +C0 en cash
        //      - buy Delta0 actions -> -Delta0 * S0 en cash
        //      => cash initial B0 = C0 - Delta0 * S0
        //
        //    Pour rester autonome, on approxime C0 via BS directement :
        const double sqrtT = std::sqrt(T);
        const double sigma2 = sigma * sigma;
        const double d1_0 = (std::log(S0 / K) + (r + 0.5 * sigma2) * T)
                            / (sigma * sqrtT);
        const double d2_0 = d1_0 - sigma * sqrtT;

        using stats::normal_cdf;
        double C0 = 0.0;
        double P0 = 0.0;
        if (option.type() == OptionType::Call) {
            C0 = S0 * normal_cdf(d1_0)
               - K * std::exp(-r * T) * normal_cdf(d2_0);
        } else {
            P0 = K * std::exp(-r * T) * normal_cdf(-d2_0)
               - S0 * normal_cdf(-d1_0);
        }

        const double optionPrice0 = (option.type() == OptionType::Call) ? C0 : P0;

        // Delta initial
        double delta = bsDelta(option, model, S, t);

        // Cash account : ce que tu as en caisse après short option + achat d'actions
        double B = optionPrice0 - delta * S;

        // 2) Boucle de hedging jusqu'à T
        for (unsigned int step = 0; step < m_cfg.nSteps; ++step) {
            // Le cash accrédite des intérêts
            B *= std::exp(r * dt);

            // Le sous-jacent évolue
            const double Z = dist(gen);
            S *= std::exp(drift + volStep * Z);

            t += dt;
            if (t > T) t = T;

            // Nouveau delta (Delta_t+)
            const double newDelta = bsDelta(option, model, S, t);

            // Rebalancement : acheter/vendre (newDelta - delta) actions au prix S
            const double dDelta = newDelta - delta;
            B -= dDelta * S;

            delta = newDelta;
        }

        // 3) À maturité : on ferme tout
        // Payoff de l'option
        double payoff = 0.0;
        if (option.type() == OptionType::Call) {
            payoff = std::max(S - K, 0.0);
        } else {
            payoff = std::max(K - S, 0.0);
        }

        // On rembourse le payoff au client (on est short l'option)
        B -= payoff;

        // On liquide la position en actions
        B += delta * S;
        delta = 0.0;

        // B = P&L final de la stratégie short option + delta hedge
        pnls[path] = B;
    }

    // Statistiques sur les P&L
    HedgingStats stats{};

    const double n = static_cast<double>(m_cfg.nPaths);
    double sum = 0.0;
    double sum2 = 0.0;
    double minv = pnls.empty() ? 0.0 : pnls[0];
    double maxv = minv;

    for (double x : pnls) {
        sum  += x;
        sum2 += x * x;
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }

    const double mean = sum / n;
    const double var  = (sum2 - n * mean * mean) / (n - 1.0);
    const double std  = std::sqrt(std::max(var, 0.0));

    stats.meanPnL = mean;
    stats.stdPnL  = std;
    stats.minPnL  = minv;
    stats.maxPnL  = maxv;

    return stats;
}
