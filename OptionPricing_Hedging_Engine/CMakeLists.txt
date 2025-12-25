cmake_minimum_required(VERSION 3.15)
project(pricing_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pricing_core
    src/core/Option.cpp
    src/core/BlackScholesModel.cpp
    src/core/Normal.cpp

    src/methods/AnalyticalPricer.cpp
    src/methods/MonteCarloPricer.cpp
    src/methods/PDESolver.cpp
    src/methods/LongstaffSchwartzPricer.cpp
    src/methods/HedgingSimulator.cpp     

    src/util/LinearSolver.cpp
    src/ui/Interface.cpp
)

target_include_directories(pricing_core PUBLIC include)

add_executable(bs_mc_pde_demo
    src/apps/bs_mc_pde_demo.cpp
)
target_link_libraries(bs_mc_pde_demo PRIVATE pricing_core)

# Nouveau binaire pour le hedging
add_executable(hedging_demo
    src/apps/hedging_demo.cpp
)
target_link_libraries(hedging_demo PRIVATE pricing_core)
