// File: src/methods/AnalyticalPricer.cpp
#include "pricing/methods/AnalyticalPricer.hpp"
#include "pricing/core/Normal.hpp"
#include <cmath>

double AnalyticalPricer::price(const Option& option,
                               const BlackScholesModel& model) const {
    const double S0    = model.spot();
    const double K     = option.strike();
    const double T     = option.maturity();
    const double r     = model.rate();
    const double sigma = model.volatility();

    const double sqrtT = std::sqrt(T);
    const double sigma2 = sigma * sigma;

    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma2) * T)
                      / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    using stats::normal_cdf;

    if (option.type() == OptionType::Call) {
        return S0 * normal_cdf(d1)
             - K * std::exp(-r * T) * normal_cdf(d2);
    } else {
        // Put : parité put-call
        return K * std::exp(-r * T) * normal_cdf(-d2)
             - S0 * normal_cdf(-d1);
    }
}
