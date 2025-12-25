// File: src/apps/bs_mc_pde_demo.cpp
#include <iostream>
#include <iomanip>

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"
#include "pricing/core/Payoff.hpp"
#include "pricing/methods/AnalyticalPricer.hpp"
#include "pricing/methods/MonteCarloPricer.hpp"
#include "pricing/methods/PDESolver.hpp"
#include "pricing/methods/LongstaffSchwartzPricer.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Paramètres de base
    const double S0    = 100.0;
    const double K     = 100.0;
    const double T     = 1.0;
    const double r     = 0.05;
    const double sigma = 0.2;

    BlackScholesModel model(S0, r, sigma);

    // ========= 1. Européen Call : Analytique / MC / PDE =========
    Option euroCall(OptionType::Call, ExerciseStyle::European, K, T);
    PayoffCall payoffCall(K);

    AnalyticalPricer analyticPricer;
    const double callAnalytic = analyticPricer.price(euroCall, model);

    std::cout << "=== European Call (S0=100, K=100, T=1, r=5%, sigma=20%) ===\n\n";
    std::cout << "Analytical Black-Scholes price: " << callAnalytic << "\n\n";

    // Monte Carlo brut
    MCConfig cfgPlain;
    cfgPlain.nPaths = 100000;
    cfgPlain.nSteps = 100;
    cfgPlain.useAntithetic = false;
    cfgPlain.useControlVariate = false;

    MonteCarloPricer mcPlain(cfgPlain);
    MCResult mcPlainRes = mcPlain.price(euroCall, model);

    // Monte Carlo avec variance reduction (antithetic + control variate)
    MCConfig cfgVR = cfgPlain;
    cfgVR.useAntithetic   = true;
    cfgVR.useControlVariate = true;

    MonteCarloPricer mcVR(cfgVR);
    MCResult mcVRRes = mcVR.price(euroCall, model);

    std::cout << "Monte Carlo (plain, N = " << cfgPlain.nPaths
              << ", steps = " << cfgPlain.nSteps << ")\n";
    std::cout << "  Price       = " << mcPlainRes.price << "\n";
    std::cout << "  Std. error  = " << mcPlainRes.stdError << "\n";
    std::cout << "  95% CI      = [" << mcPlainRes.ciLower
              << ", " << mcPlainRes.ciUpper << "]\n\n";

    std::cout << "Monte Carlo (variance reduction: antithetic + control variate)\n";
    std::cout << "  Price       = " << mcVRRes.price << "\n";
    std::cout << "  Std. error  = " << mcVRRes.stdError << "\n";
    std::cout << "  95% CI      = [" << mcVRRes.ciLower
              << ", " << mcVRRes.ciUpper << "]\n\n";

    // PDE (européen)
    const double S_max = 4.0 * std::max(S0, K);
    const std::size_t N = 200;
    const std::size_t M = 2000;

    PDESolver pdeSolver(T, r, sigma, S_max, N, M);
    pdeSolver.precomputeMatrices();
    PricingResults pdeEuroCall = pdeSolver.solve(payoffCall, S0, false);

    std::cout << "PDE Crank-Nicolson (European call)\n";
    std::cout << "  Price       = " << pdeEuroCall.price << "\n";
    std::cout << "  Delta       = " << pdeEuroCall.delta << "\n";
    std::cout << "  Gamma       = " << pdeEuroCall.gamma << "\n\n";

    std::cout << "=== Comparison (European call) ===\n";
    auto absErr = [callAnalytic](double p) { return std::abs(p - callAnalytic); };

    std::cout << std::left
              << std::setw(20) << "Method"
              << std::setw(15) << "Price"
              << std::setw(20) << "Abs error vs analytic"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::setw(20) << "Analytic"
              << std::setw(15) << callAnalytic
              << std::setw(20) << 0.0 << "\n";

    std::cout << std::setw(20) << "MC (plain)"
              << std::setw(15) << mcPlainRes.price
              << std::setw(20) << absErr(mcPlainRes.price) << "\n";

    std::cout << std::setw(20) << "MC (VR)"
              << std::setw(15) << mcVRRes.price
              << std::setw(20) << absErr(mcVRRes.price) << "\n";

    std::cout << std::setw(20) << "PDE (CN)"
              << std::setw(15) << pdeEuroCall.price
              << std::setw(20) << absErr(pdeEuroCall.price) << "\n\n";

    // ========= 2. American Put : PDE obstacle vs LSMC =========
    std::cout << "\n=== American Put (PDE obstacle vs LSMC) ===\n\n";

    Option amerPut(OptionType::Put, ExerciseStyle::American, K, T);
    PayoffPut payoffPut(K);

    // PDE American (obstacle)
    PricingResults pdeAmerPut = pdeSolver.solve(payoffPut, S0, true);

    std::cout << "PDE (American put)\n";
    std::cout << "  Price  = " << pdeAmerPut.price << "\n";
    std::cout << "  Delta  = " << pdeAmerPut.delta << "\n";
    std::cout << "  Gamma  = " << pdeAmerPut.gamma << "\n\n";

    // LSMC
    LongstaffSchwartzPricer lsmc(50000, 50); // 50k paths, 50 exercise dates
    LSMCResult lsmcRes = lsmc.price(amerPut, model, payoffPut);

    std::cout << "Longstaff-Schwartz MC (American put)\n";
    std::cout << "  Price       = " << lsmcRes.price << "\n";
    std::cout << "  Std. error  = " << lsmcRes.stdError << "\n\n";

    std::cout << "=== Comparison (American put) ===\n";
    std::cout << std::left
              << std::setw(20) << "Method"
              << std::setw(15) << "Price"
              << "\n";
    std::cout << std::string(35, '-') << "\n";

    std::cout << std::setw(20) << "PDE (obstacle)"
              << std::setw(15) << pdeAmerPut.price << "\n";
    std::cout << std::setw(20) << "LSMC"
              << std::setw(15) << lsmcRes.price << "\n";

    return 0;
}
