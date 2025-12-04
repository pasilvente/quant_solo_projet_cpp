#include <iostream>
#include <vector>
#include <iomanip>
#include "Option.hpp"
#include "BlackScholesModel.hpp"
#include "AnalyticalPricer.hpp"
#include "MonteCarloPricer.hpp"

int main() {
    double S0    = 100.0;
    double K     = 100.0;
    double T     = 1.0;      // 1 an
    double r     = 0.05;     // 5 %
    double sigma = 0.2;      // 20 %

    Option call(OptionType::Call, K, T);
    Option put(OptionType::Put, K, T);
    BlackScholesModel model(S0, r, sigma);

    AnalyticalPricer analyticPricer;

    double callAnalytic = analyticPricer.price(call, model);
    double putAnalytic  = analyticPricer.price(put, model);

    std::cout << "=== Black-Scholes analytical pricing ===\n";
    std::cout << "Call (analytic) = " << callAnalytic << "\n";
    std::cout << "Put  (analytic) = " << putAnalytic  << "\n\n";

    // Monte Carlo de base (N = 100000, steps = 100)
    unsigned int nPaths = 100000;
    unsigned int nSteps = 100;

    MonteCarloPricer mcPricer(nPaths, nSteps);

    auto callMC = mcPricer.price(call, model);
    auto putMC  = mcPricer.price(put, model);

    std::cout << "=== Monte Carlo pricing (single configuration) ===\n";
    std::cout << "Parameters: N = " << nPaths << ", steps = " << nSteps << "\n\n";

    std::cout << "Call (MC) price       = " << callMC.price << "\n";
    std::cout << "Call (MC) std. error  = " << callMC.stdError << "\n";
    std::cout << "Call (MC) 95% CI      = [" << callMC.ciLower
              << ", " << callMC.ciUpper << "]\n\n";

    std::cout << "Put  (MC) price       = " << putMC.price << "\n";
    std::cout << "Put  (MC) std. error  = " << putMC.stdError << "\n";
    std::cout << "Put  (MC) 95% CI      = [" << putMC.ciLower
              << ", " << putMC.ciUpper << "]\n\n";

    // ============================
    // Étude de convergence
    // ============================
    std::cout << "=== Convergence study for the call option ===\n";

    std::vector<unsigned int> Ns = {1000, 5000, 10000, 50000, 100000};
    nSteps = 100;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Analytical call price = " << callAnalytic << "\n\n";

    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(15) << "MC price"
              << std::setw(15) << "Abs error"
              << std::setw(15) << "Std error"
              << std::setw(30) << "95% CI"
              << "\n";

    std::cout << std::string(10 + 15 + 15 + 15 + 30, '-') << "\n";

    for (auto N : Ns) {
        MonteCarloPricer pricerN(N, nSteps);
        auto res = pricerN.price(call, model);

        double absError = std::abs(res.price - callAnalytic);

        std::cout << std::setw(10) << N
                  << std::setw(15) << res.price
                  << std::setw(15) << absError
                  << std::setw(15) << res.stdError;

        std::cout << "[" << res.ciLower << ", " << res.ciUpper << "]\n";
    }

    return 0;
}
