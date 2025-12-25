// File: src/apps/hedging_demo.cpp
#include <iostream>
#include <iomanip>

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"
#include "pricing/methods/HedgingSimulator.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double S0    = 100.0;
    const double K     = 100.0;
    const double T     = 1.0;
    const double r     = 0.05;
    const double sigma = 0.2;

    BlackScholesModel model(S0, r, sigma);
    Option euroCall(OptionType::Call, ExerciseStyle::European, K, T);

    std::cout << "=== Delta-hedging P&L for a European call ===\n";
    std::cout << "S0=" << S0 << ", K=" << K << ", T=1, r=5%, sigma=20%\n\n";

    // Hedging "quotidien" ~252 rebalancements
    HedgingConfig cfgDaily;
    cfgDaily.nPaths = 20000;
    cfgDaily.nSteps = 252;

    HedgingSimulator simDaily(cfgDaily);
    HedgingStats statsDaily = simDaily.simulateDeltaHedge(euroCall, model);

    std::cout << "Daily hedging (252 steps):\n";
    std::cout << "  mean P&L = " << statsDaily.meanPnL << "\n";
    std::cout << "  std  P&L = " << statsDaily.stdPnL  << "\n";
    std::cout << "  min  P&L = " << statsDaily.minPnL  << "\n";
    std::cout << "  max  P&L = " << statsDaily.maxPnL  << "\n\n";

    // Hedging plus rare, ex: hebdo ~52 rebalancements
    HedgingConfig cfgWeekly;
    cfgWeekly.nPaths = 20000;
    cfgWeekly.nSteps = 52;

    HedgingSimulator simWeekly(cfgWeekly);
    HedgingStats statsWeekly = simWeekly.simulateDeltaHedge(euroCall, model);

    std::cout << "Weekly hedging (52 steps):\n";
    std::cout << "  mean P&L = " << statsWeekly.meanPnL << "\n";
    std::cout << "  std  P&L = " << statsWeekly.stdPnL  << "\n";
    std::cout << "  min  P&L = " << statsWeekly.minPnL  << "\n";
    std::cout << "  max  P&L = " << statsWeekly.maxPnL  << "\n";

    return 0;
}
