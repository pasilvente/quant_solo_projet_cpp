// File: src/core/BlackScholesModel.cpp
#include "pricing/core/BlackScholesModel.hpp"

BlackScholesModel::BlackScholesModel(double spot,
                                     double rate,
                                     double volatility)
    : m_spot(spot),
      m_rate(rate),
      m_vol(volatility) {}
