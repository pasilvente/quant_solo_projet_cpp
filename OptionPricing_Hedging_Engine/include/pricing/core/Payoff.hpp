// File: include/pricing/core/Payoff.hpp
#pragma once

#include <algorithm>

/**
 * @brief Interface abstraite représentant le payoff d'une option européenne.
 *
 * Cette classe définit le "contrat" d'un produit : pour un spot S donné,
 * combien l'option rapporte à maturité.
 *
 * Exemple d'utilisation :
 *   PayoffCall callPayoff(100.0);
 *   double value = callPayoff(105.0); // max(105 - 100, 0) = 5
 */
class Payoff {
public:
    /// Destructeur virtuel pour permettre une destruction polymorphique correcte.
    virtual ~Payoff() = default;

    /**
     * @brief Évalue le payoff pour un prix spot donné.
     * @param spot Prix du sous-jacent S_T à maturité.
     * @return Valeur du payoff à maturité.
     */
    virtual double operator()(double spot) const = 0;
};

/**
 * @brief Payoff d'un call européen : max(S_T - K, 0).
 */
class PayoffCall : public Payoff {
private:
    double m_strike; ///< Prix d'exercice K

public:
    explicit PayoffCall(double strike)
        : m_strike(strike) {}

    double operator()(double spot) const override {
        return std::max(spot - m_strike, 0.0);
    }
};

/**
 * @brief Payoff d'un put européen : max(K - S_T, 0).
 */
class PayoffPut : public Payoff {
private:
    double m_strike; ///< Prix d'exercice K

public:
    explicit PayoffPut(double strike)
        : m_strike(strike) {}

    double operator()(double spot) const override {
        return std::max(m_strike - spot, 0.0);
    }
};
