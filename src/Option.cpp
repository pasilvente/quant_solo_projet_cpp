#include "Option.hpp"

Option::Option(OptionType type, double strike, double maturity)
    : m_type(type), m_strike(strike), m_maturity(maturity) {}
