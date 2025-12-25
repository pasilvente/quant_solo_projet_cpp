# Option Pricing under Black–Scholes in C++

Small personal project: pricing European options under the Black–Scholes model, with both **closed-form formulas** and **Monte Carlo simulation**.

## 1. Model

We assume the underlying follows a **Geometric Brownian Motion**:

\[
dS_t = r S_t\, dt + \sigma S_t\, dW_t,
\]

with constant interest rate \(r\) and volatility \(\sigma\).

Under the risk-neutral measure, the price of a European call with strike \(K\) and maturity \(T\) is

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2),
\]

with

\[
d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\quad
d_2 = d_1 - \sigma\sqrt{T}.
\]

The put price is obtained by put–call parity.

## 2. Project structure

- `include/` – headers (`Option`, `BlackScholesModel`, `AnalyticalPricer`, `MonteCarloPricer`, `Normal`)
- `src/` – implementation files
- `CMakeLists.txt` – CMake configuration
- `README.md` – this file

Main classes:

- `AnalyticalPricer` – closed-form Black–Scholes prices (call/put)
- `MonteCarloPricer` – Monte Carlo pricing with
  - Geometric Brownian Motion paths
  - discounted payoffs
  - empirical variance and **95% confidence interval**

## 3. Build & run

Requirements:

- CMake
- A C++17 compiler (MSVC, g++, clang…)

Build with CMake:

```bash
cmake -B build
cmake --build build

## 5. Sample output

```text
=== Black-Scholes analytical pricing ===
Call (analytic) = 10.4506
Put  (analytic) = 5.57353

=== Monte Carlo pricing (single configuration) ===
Parameters: N = 100000, steps = 100

Call (MC) price       = 10.4738
Call (MC) std. error  = 0.0467053
Call (MC) 95% CI      = [10.3822, 10.5653]

Put  (MC) price       = 5.57873
Put  (MC) std. error  = 0.0274573
Put  (MC) 95% CI      = [5.52491, 5.63254]

=== Convergence study for the call option ===
Analytical call price = 10.450584

N         MC price       Abs error      Std error      95% CI
-------------------------------------------------------------------------------------
1000      10.877151      0.426568       0.471605       [9.952806, 11.801496]
5000      10.313047      0.137537       0.204341       [9.912539, 10.713555]
10000     10.221770      0.228813       0.144762       [9.938038, 10.505503]
50000     10.427498      0.023085       0.066130       [10.297884, 10.557112]
100000    10.546539      0.095955       0.046899       [10.454616, 10.638461]

