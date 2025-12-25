// File: include/pricing/ui/Interface.hpp
#pragma once

#include <cstddef>

/**
 * @brief Interface simple d'entrée utilisateur (console) pour paramétrer une option.
 *
 * Cette classe encapsule :
 *  - les paramètres financiers (S0, K, T, r, sigma),
 *  - les paramètres de la grille pour le solveur PDE (S_max, N, M, etc.),
 *  - le type d'option (call ou put).
 *
 * Méthode typique d'utilisation :
 *   Interface ui;
 *   ui.askParameters();
 *   double S0   = ui.getS0();
 *   double K    = ui.getK();
 *   double T    = ui.getT();
 *   // etc.
 */
class Interface {
private:
    // Paramètres financiers
    double m_S0     = 0.0; ///< Prix du sous-jacent (spot)
    double m_K      = 0.0; ///< Strike
    double m_T      = 0.0; ///< Maturité (en années)
    double m_r      = 0.0; ///< Taux d'intérêt sans risque
    double m_sigma  = 0.0; ///< Volatilité implicite

    // Paramètres de grille / solver PDE
    double m_S_max  = 0.0; ///< Borne supérieure pour S sur la grille
    double m_theta  = 0.5; ///< Paramètre temporel (0.5 = Crank–Nicolson)
    std::size_t m_M = 0;   ///< Nombre de pas de temps
    std::size_t m_N = 0;   ///< Nombre de points d'espace

    bool m_isCall   = true; ///< true = Call, false = Put

public:
    /// Constructeur par défaut
    Interface() = default;

    /**
     * @brief Interagit avec l'utilisateur (console) pour saisir tous les paramètres.
     *
     * Cette méthode peut :
     *  - afficher des messages explicatifs,
     *  - valider les entrées,
     *  - éventuellement proposer des valeurs par défaut.
     *
     * L'implémentation sera faite dans le .cpp.
     */
    void askParameters();

    // Getters pour exposer les paramètres au reste du code

    double getS0() const { return m_S0; }
    double getK() const { return m_K; }
    double getT() const { return m_T; }
    double getR() const { return m_r; }
    double getSigma() const { return m_sigma; }

    double getS_max() const { return m_S_max; }
    std::size_t getM() const { return m_M; }
    std::size_t getN() const { return m_N; }

    double getTheta() const { return m_theta; }
    bool getIsCall() const { return m_isCall; }
};
