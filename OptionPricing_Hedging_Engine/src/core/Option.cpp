// File: src/core/Option.cpp
#include "pricing/core/Option.hpp"

Option::Option(OptionType type,
               ExerciseStyle style,
               double strike,
               double maturity)
    : m_type(type),
      m_style(style),
      m_strike(strike),
      m_maturity(maturity) {}
