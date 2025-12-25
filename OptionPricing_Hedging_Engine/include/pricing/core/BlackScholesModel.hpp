#pragma once

class BlackScholesModel {
public:
    BlackScholesModel(double spot, double rate, double volatility);

    double spot() const { return m_spot; }
    double rate() const { return m_rate; }         // taux sans risque
    double volatility() const { return m_vol; }    // sigma

private:
    double m_spot;
    double m_rate;
    double m_vol;
};
