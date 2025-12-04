#include "AnalyticalPricer.hpp"
#include "Normal.hpp"
#include <cmath>

double AnalyticalPricer::price(const Option& option, const BlackScholesModel& model) const {
    double S0 = model.spot();
    double K  = option.strike();
    double T  = option.maturity();
    double r  = model.rate();
    double sigma = model.volatility();

    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    using stats::normal_cdf;

    if (option.type() == OptionType::Call) {
        return S0 * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    } else { // Put
        return K * std::exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1);
    }
}
