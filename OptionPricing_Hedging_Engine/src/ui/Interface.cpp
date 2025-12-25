// File: src/ui/Interface.cpp
#include "pricing/ui/Interface.hpp"

#include <iostream>
#include <limits>

namespace {

    template <typename T>
    void readPositive(const std::string& label, T& value) {
        while (true) {
            std::cout << label;
            if (std::cin >> value && value > static_cast<T>(0)) {
                break;
            }
            std::cout << "Please enter a positive number.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

} // namespace

void Interface::askParameters() {
    std::cout << "=== PDE Option Pricer Parameters ===\n";

    readPositive("Spot S0: ", m_S0);
    readPositive("Strike K: ", m_K);
    readPositive("Maturity T (years): ", m_T);

    std::cout << "Risk-free rate r (e.g. 0.05 for 5%): ";
    while (!(std::cin >> m_r)) {
        std::cout << "Please enter a valid number.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Risk-free rate r: ";
    }

    readPositive("Volatility sigma (e.g. 0.2 for 20%): ", m_sigma);

    // Paramètres de grille
    std::cout << "\n=== Grid parameters ===\n";

    readPositive("S_max (upper bound in S): ", m_S_max);

    {
        int tmp = 0;
        readPositive("Number of space steps N (>= 50 recommended): ", tmp);
        m_N = static_cast<std::size_t>(tmp);
    }

    {
        int tmp = 0;
        readPositive("Number of time steps M (>= 100 recommended): ", tmp);
        m_M = static_cast<std::size_t>(tmp);
    }

    m_theta = 0.5; // Crank–Nicolson par défaut

    // Type d'option
    std::cout << "\nOption type (1 = Call, 2 = Put): ";
    int type = 1;
    while (true) {
        if (std::cin >> type && (type == 1 || type == 2)) {
            break;
        }
        std::cout << "Please enter 1 (Call) or 2 (Put): ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    m_isCall = (type == 1);
}
