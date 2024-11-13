#pragma once
#include <iostream>
#include <filesystem>
#include <string.h>
#include <matplot/matplot.h>
#include <single_include/nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <random>

#include <armadillo>
#define MLPACK_ENABLE_ANN_SERIALIZATION
#define CEREAL_THREAD_SAFE 1
#include <mlpack.hpp>