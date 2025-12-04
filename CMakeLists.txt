cmake_minimum_required(VERSION 3.15)
project(option_pricing_bs_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(option_pricing_main
    src/main.cpp
    src/Option.cpp
    src/BlackScholesModel.cpp
    src/AnalyticalPricer.cpp
    src/Normal.cpp
    src/MonteCarloPricer.cpp      # <-- ajouté
)

target_include_directories(option_pricing_main PUBLIC include)
