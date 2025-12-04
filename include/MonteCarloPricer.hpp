#pragma once

#include "Option.hpp"
#include "BlackScholesModel.hpp"

class MonteCarloPricer {
public:
    struct Result {
        double price;
        double stdError;
        double ciLower;
        double ciUpper;
    };

    MonteCarloPricer(unsigned int nPaths, unsigned int nSteps);

    Result price(const Option& option, const BlackScholesModel& model) const;

private:
    unsigned int m_nPaths;
    unsigned int m_nSteps;
};
