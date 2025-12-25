// File: include/pricing/core/Option.hpp
#pragma once

/**
 * Type de payoff : call ou put.
 */
enum class OptionType {
    Call,
    Put
};

/**
 * Style d'exercice : européen ou américain.
 */
enum class ExerciseStyle {
    European,
    American
};

/**
 * @brief Représente une option vanille (call/put, euro/américaine).
 *
 * On stocke simplement le type, le style, le strike et la maturité.
 */
class Option {
public:
    Option(OptionType type,
           ExerciseStyle style,
           double strike,
           double maturity);

    OptionType type() const { return m_type; }
    ExerciseStyle style() const { return m_style; }
    double strike() const { return m_strike; }
    double maturity() const { return m_maturity; }

private:
    OptionType      m_type;
    ExerciseStyle   m_style;
    double          m_strike;
    double          m_maturity; // en années
};
