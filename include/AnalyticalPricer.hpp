#pragma once

#include "Option.hpp"
#include "BlackScholesModel.hpp"

class AnalyticalPricer {
public:
    double price(const Option& option, const BlackScholesModel& model) const;
};
