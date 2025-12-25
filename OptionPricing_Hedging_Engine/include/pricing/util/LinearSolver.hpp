// File: include/pricing/util/LinearSolver.hpp
#pragma once

#include <vector>
#include <stdexcept>

namespace Solver {

    /**
     * @brief Résout un système linéaire tridiagonal A x = d via l'algorithme de Thomas.
     *
     * L'algorithme de Thomas est une version spécialisée du pivot de Gauss optimisée
     * pour les matrices tridiagonales. Il exploite la structure particulière de A
     * pour obtenir une complexité linéaire O(N), contre O(N^3) pour une élimination
     * gaussienne générale.
     *
     * La matrice A est représentée par trois vecteurs :
     *  - a : diagonale inférieure (lower diagonal), taille N
     *  - b : diagonale principale (main diagonal), taille N
     *  - c : diagonale supérieure (upper diagonal), taille N
     *
     * Conventions :
     *  - a[0] n'est pas utilisé (pas d'élément sous la première ligne).
     *  - c[N-1] n'est pas utilisé (pas d'élément au-dessus de la dernière ligne).
     *
     * @param a Vecteur de la diagonale inférieure (taille N).
     * @param b Vecteur de la diagonale principale (taille N).
     * @param c Vecteur de la diagonale supérieure (taille N).
     * @param d Vecteur du second membre (taille N).
     * @param x Vecteur résultat (taille N). Il sera rempli par la fonction.
     *
     * @throws std::invalid_argument si les tailles des vecteurs ne sont pas cohérentes.
     * @throws std::runtime_error si le système est singulier (division par zéro).
     */
    void thomasAlgorithm(const std::vector<double>& a,
                         const std::vector<double>& b,
                         const std::vector<double>& c,
                         const std::vector<double>& d,
                         std::vector<double>& x);

} // namespace Solver
