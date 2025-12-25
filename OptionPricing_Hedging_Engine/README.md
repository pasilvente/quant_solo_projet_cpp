# C++ Option Pricing & Hedging Engine (Black–Scholes)

This project is a small C++ pricing and risk engine built around the Black–Scholes model. I wrote it to consolidate the “classic” tools used on derivatives desks:

- closed-form Black–Scholes pricing,
- Monte Carlo pricing with variance reduction,
- finite-difference PDE methods (Crank–Nicolson),
- American option pricing (PDE obstacle & Longstaff–Schwartz),
- and delta-hedging P&L simulations.

It is not meant to be a full production library, but a compact, readable implementation that shows I understand both the **maths** and the **numerics** behind standard option pricing and hedging.

---

## 1. Features

### Pricing under Black–Scholes

- **European options**
  - Closed-form Black–Scholes formula (calls and puts)
  - Monte Carlo pricing
    - plain Monte Carlo
    - **variance reduction**:
      - antithetic variates  
      - control variate based on the discounted underlying \(e^{-rT} S_T\)
  - PDE finite-difference solver
    - Crank–Nicolson scheme
    - tridiagonal systems solved with the Thomas algorithm
    - Delta and Gamma estimated from the PDE grid

- **American options**
  - PDE with **obstacle condition** (early exercise handled via projection on the payoff at each time step)
  - **Longstaff–Schwartz Monte Carlo (LSMC)** with polynomial basis \([1, S, S^2]\)

### Risk & Hedging

- **Delta-hedging P&L simulation** under Black–Scholes:
  - simulate many GBM price paths,
  - sell one call at the Black–Scholes price,
  - dynamically hedge with the analytical delta,
  - track the P&L of the strategy:
    \[
      \text{P\&L} = \text{cash account} + \Delta_T S_T - \text{option payoff}.
    \]
  - compare the P&L distribution for different rebalancing frequencies  
    (e.g. daily vs weekly delta-hedging).

---

## 2. Mathematical framework

The underlying asset \(S_t\) follows a geometric Brownian motion under the risk-neutral measure:

\[
dS_t = r S_t\,dt + \sigma S_t\,dW_t,
\]

with constant risk-free rate \(r\) and volatility \(\sigma\).

The price \(V(t, S)\) of a European derivative under Black–Scholes satisfies the PDE:

\[
\frac{\partial V}{\partial t}
+ \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S}
- r V = 0,
\]

with terminal condition at maturity \(T\):

\[
V(T, S) = \text{payoff}(S).
\]

- For **European options**, the PDE is solved with a Crank–Nicolson finite-difference scheme on a uniform grid in \(S\) and \(t\).
- For **American options**, an **obstacle condition** is enforced at each time step:
  \[
  V(t, S) \ge \text{payoff}(S),
  \]
  by projecting the numerical solution onto the payoff whenever early exercise is optimal.

The **Longstaff–Schwartz** method provides an alternative Monte Carlo approach for American options:
- simulate many paths,
- work backwards in time,
- on in-the-money paths, regress the discounted future payoff on basis functions of \(S_t\) to approximate the continuation value,
- exercise when the intrinsic value exceeds the estimated continuation value.

For hedging, I use the **analytical Black–Scholes delta**:

\[
\Delta_{\text{call}} = N(d_1), \quad
\Delta_{\text{put}} = N(d_1) - 1,
\]

and simulate the P&L of a short option position with discrete-time delta rebalancing.

---

## 3. Project structure

```text
OptionPricing_Hedging_Engine/
  include/
    pricing/
      core/
        Option.hpp              # Option type (call/put) and style (European/American)
        BlackScholesModel.hpp   # S0, r, sigma
        Payoff.hpp              # Payoff interface + European call/put payoffs
        Normal.hpp              # Normal CDF / PDF utilities
      methods/
        AnalyticalPricer.hpp            # Closed-form Black–Scholes pricing
        MonteCarloPricer.hpp            # MC pricing (with variance reduction)
        PDESolver.hpp                   # Crank–Nicolson PDE solver (Euro & American)
        LongstaffSchwartzPricer.hpp     # LSMC for American options
        HedgingSimulator.hpp            # Delta-hedging P&L simulator under BS
      util/
        LinearSolver.hpp        # Thomas algorithm for tridiagonal systems
      ui/
        Interface.hpp           # (Optional) user input interface
  src/
    core/
      Option.cpp
      BlackScholesModel.cpp
      Normal.cpp
    methods/
      AnalyticalPricer.cpp
      MonteCarloPricer.cpp
      PDESolver.cpp
      LongstaffSchwartzPricer.cpp
      HedgingSimulator.cpp
    util/
      LinearSolver.cpp
    ui/
      Interface.cpp
    apps/
      bs_mc_pde_demo.cpp        # Pricing demo: analytic vs MC vs PDE vs American (LSMC)
      hedging_demo.cpp          # Delta-hedging P&L demo (daily vs weekly rebalancing)
  CMakeLists.txt
  README.md


---

## 4. Building

The project is built with CMake and requires a C++17 compiler.

Typical out-of-source build:

cmake -B build
cmake --build build

This creates the executables in build/:

bs_mc_pde_demo

hedging_demo

On Windows / MSVC, the project can also be built and run via the CMake Tools extension in VS Code.

---

## 5. Usage

## 5.1 Pricing & comparison demo

The bs_mc_pde_demo executable prices a European call and an American put, and compares the different methods.

For an European call (S0=100, K=100, T=1, r=5%, sigma=20%) :

=== European Call (S0=100, K=100, T=1, r=5%, sigma=20%) ===

Analytical Black-Scholes price: 10.450584

Monte Carlo (plain, N = 100000, steps = 100)
  Price       = 10.397115
  Std. error  = 0.046535
  95% CI      = [10.305907, 10.488322]

Monte Carlo (variance reduction: antithetic + control variate)
  Price       = 10.465984
  Std. error  = 0.012564
  95% CI      = [10.441359, 10.490609]

PDE Crank-Nicolson (European call)
  Price       = 10.440685
  Delta       = 0.598026
  Gamma       = 0.019764

=== Comparison (European call) ===
Method              Price          Abs error vs analytic

Analytic            10.450584      0.000000
MC (plain)          10.397115      0.053469
MC (VR)             10.465984      0.015400
PDE (CN)            10.440685      0.009898


This output shows: the plain MC estimate is within about one standard deviation of the analytical price, variance reduction significantly tightens the the confidence interval, the PDE Crank–Nicolson solution matches the analytical result up to a small discretisation error.

For an American put (PDE obstacle vs Longstaff–Schwartz) :

=== American Put (PDE obstacle vs LSMC) ===

PDE (American put)
  Price  = 6.078800
  Delta  = -0.459280
  Gamma  = 0.024761

Longstaff–Schwartz MC (American put)
  Price       = 6.068449
  Std. error  = 0.032425

=== Comparison (American put) ===
Method              Price

PDE (obstacle)      6.078800
LSMC                6.068449


Both methods produce very similar prices (well within one MC standard error), which is a good consistency check between the PDE and LSMC implementations.

## 5.2 Delta-hedging P&L demo

The hedging_demo executable simulates the P&L of a short European call hedged with the analytical delta, under Black–Scholes dynamics, for different rebalancing frequencies.

Example output:

=== Delta-hedging P&L for a European call ===
S0=100.000000, K=100.000000, T=1, r=5%, sigma=20%

Daily hedging (252 steps):
  mean P&L = 0.004418
  std  P&L = 0.440580
  min  P&L = -3.477283
  max  P&L = 2.480433

Weekly hedging (52 steps):
  mean P&L = -0.004096
  std  P&L = 0.960237
  min  P&L = -6.336752
  max  P&L = 4.707606


Interpretation: in a Black–Scholes world priced and hedged with the correct model, the mean P&L of the delta-hedged short option is close to zero (as expected). The P&L distribution is much tighter with daily hedging than with weekly hedging, which illustrates the impact of discrete rebalancing vs the continuous-time idealised theory.

---

## 6. Possible extensions

If I had more time, natural extensions of this engine would be: calibration of Black–Scholes implied volatility from market quotes, convergence studies (PDE grid refinement, MC paths vs error), additional payoffs (digitals, barriers) built on the same infrastructure.

For the purposes of this project, I focused on implementing and cross-checking the core building blocks that appear again and again in practical derivatives pricing and risk management.