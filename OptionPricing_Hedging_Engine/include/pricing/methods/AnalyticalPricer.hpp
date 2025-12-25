#pragma once

#include "pricing/core/Option.hpp"
#include "pricing/core/BlackScholesModel.hpp"

class AnalyticalPricer {
public:
    double price(const Option& option, const BlackScholesModel& model) const;
};
