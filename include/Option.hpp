#pragma once

enum class OptionType {
    Call,
    Put
};

class Option {
public:
    Option(OptionType type, double strike, double maturity);

    OptionType type() const { return m_type; }
    double strike() const { return m_strike; }
    double maturity() const { return m_maturity; }

private:
    OptionType m_type;
    double m_strike;
    double m_maturity; // en années
};
