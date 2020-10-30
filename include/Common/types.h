#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <stdexcept>
#include <sstream>
#include <limits>

using real = float;
using real_vec = std::vector<real>;
using real_matrix = std::vector<real_vec>;
using real_3d_matrix = std::vector<real_matrix>;

#define EPSILON 0.001f

#define THROW_ERROR(X) std::stringstream ss; \
                        ss << X; \
                        throw std::invalid_argument(std::string(std::string(__FILE__) \
                        + " | func:" + std::string(__FUNCTION__) \
                        + " | line:" + std::to_string(__LINE__) \
                        + " > " + ss.str()))

// Fix with std::optional
#define NONE_REAL   std::numeric_limits<real>::quiet_NaN()

namespace InitializationFunction {
    struct ALL_ZERO
    {

    };

    struct RANDOM_NORMAL
    {
        real mean = 0.0;
        real stddev = 1.0;
        unsigned long seed = 0; // Fix with std::optional
    };

    struct RANDOM_UNIFORM
    {
        unsigned long seed = 0; // Fix with std::optional
    };
}

namespace CostFunction {
    struct DIFFERENCE
    {

    };

    struct CROSS_ENTROPY
    {

    };

    struct SQUARE_DIFFERENCE
    {

    };
}

namespace ActivationFunction {
    struct RELU
    {
        real alpha = 0.0;
        real max_value = NONE_REAL;
        real threshold = 0.0;
    };

    struct SIGMOID
    {

    };

    struct TANH
    {

    };
}

namespace OptimizeFunction {
    struct TEST
    {

    };

    struct BACKPROP
    {
        real learningRate = 0.01;
        real momentum = 0.0;
    };
}


#endif // TYPES_H
