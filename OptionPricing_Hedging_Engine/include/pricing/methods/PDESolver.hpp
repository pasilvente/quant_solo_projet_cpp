// File: include/pricing/methods/PDESolver.hpp
#pragma once

#include "pricing/core/Payoff.hpp"
#include <vector>
#include <cstddef>

/**
 * @brief Structure de sortie d'un solver PDE : prix + sensibilités.
 *
 * Encapsule le résultat complet du pricing :
 *  - price : V(S0, 0)
 *  - delta : dV/dS au spot initial
 *  - gamma : d²V/dS² au spot initial
 *  - theta : dV/dt (ici laissé à 0 par simplicité)
 */
struct PricingResults {
    double price  = 0.0;
    double delta  = 0.0;
    double gamma  = 0.0;
    double theta  = 0.0;
};

/**
 * @brief Solveur PDE pour l'équation de Black–Scholes (schéma aux différences finies).
 *
 * Résout la PDE de Black–Scholes pour une option européenne
 * ou américaine (via obstacle) sous forme :
 *
 *   ∂V/∂t + 1/2 σ² S² ∂²V/∂S² + r S ∂V/∂S - r V = 0
 *
 * Schéma : Crank–Nicolson implicite, avec résolutions tridiagonales
 * (Thomas algorithm). On discrétise S ∈ [0, S_max], t ∈ [0, T].
 */
class PDESolver {
private:
    // Paramètres du modèle Black–Scholes
    double m_T;      ///< Maturité
    double m_r;      ///< Taux sans risque
    double m_sigma;  ///< Volatilité

    // Paramètres de la grille
    double m_S_max;  ///< Borne supérieure en S
    std::size_t m_N; ///< Nombre de points d'espace (S)
    std::size_t m_M; ///< Nombre de points de temps (t)

    // Pas de discrétisation
    double m_dt;     ///< Pas de temps
    double m_dS;     ///< Pas d'espace (si on travaille directement en S)

    // Coefficients des matrices (pré-calculés)
    std::vector<double> m_B_lower, m_B_diag, m_B_upper; ///< Matrice explicite (droite)
    std::vector<double> m_A_lower, m_A_diag, m_A_upper; ///< Matrice implicite (gauche)

public:
    /**
     * @brief Constructeur principal du solveur PDE.
     *
     * @param T       Maturité.
     * @param r       Taux sans risque.
     * @param sigma   Volatilité.
     * @param S_max   Borne supérieure de la grille en S.
     * @param N       Nombre de points d'espace (>= 3 recommandé).
     * @param M       Nombre de pas de temps (>= 1).
     */
    PDESolver(double T,
              double r,
              double sigma,
              double S_max,
              std::size_t N,
              std::size_t M);

    /**
     * @brief Pré-calcul des matrices de discrétisation Crank–Nicolson.
     *
     * Construit les vecteurs m_A_* et m_B_* utilisés à chaque pas de temps.
     * À appeler une fois avant la boucle principale de temps.
     */
    void precomputeMatrices();

    /**
     * @brief Résout la PDE pour un payoff donné et renvoie prix + Greeks au spot S0.
     *
     * @param payoff     Payoff de l'option (call, put, etc.).
     * @param S0         Spot initial pour lequel on veut V(S0, 0).
     * @param isAmerican true pour option américaine (obstacle V ≥ payoff),
     *                   false pour option européenne standard.
     *
     * @return PricingResults contenant le prix et les sensibilités.
     */
    PricingResults solve(const Payoff& payoff,
                         double S0,
                         bool isAmerican = false);
};
