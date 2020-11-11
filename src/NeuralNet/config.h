#ifndef CONFIG_H
#define CONFIG_H

#include <Common/types.h>


namespace NeuralNet
{


class Context;


struct Config
{
    std::vector<size_t> sizeVec;
    unsigned int batchSize;
    unsigned int printInterval;

    // Built-in initialization functions
    enum class InitFuncType
    {
        CUSTOM,
        ALL_ZERO,
        RANDOM,
        UNIFORM
    };

    // Built-in activation functions
    enum class ActivFuncType
    {
        CUSTOM,
        RELU,
        SIGMOID,
        TANH
    };

    // Built-in cost functions
    enum class CostFuncType
    {
        CUSTOM,
        DIFFERENCE,
        CROSS_ENTROPY,
        SQUARE_DIFFERENCE
    };

    // Built-in optimization functions
    enum class OptFuncType
    {
        CUSTOM,
        TEST,
        BACKPROP
    };

    // Configuration for user-definable functions
    struct InitFuncConfig
    {
        InitializationFunction::ALL_ZERO all_zero;
        InitializationFunction::RANDOM_NORMAL random;
        InitializationFunction::RANDOM_UNIFORM uniform;
    };

    struct ActivFuncConfig
    {
        ActivationFunction::RELU relu;
        ActivationFunction::SIGMOID sigmoid;
        ActivationFunction::TANH tanh;
    };

    struct CostFuncConfig
    {
        CostFunction::DIFFERENCE diff;
        CostFunction::SQUARE_DIFFERENCE sq_diff;
        CostFunction::CROSS_ENTROPY x_ntrp;
    };

    struct OptFuncConfig
    {
        OptimizeFunction::TEST test;
        OptimizeFunction::BACKPROP backprop;
    };

    // Internal structures for user-definable functions
    struct InitFunc
    {
        InitFuncConfig cfg;
        InitFuncType type;
        void (*ptr)(Context&);
    } initFunc;

    struct ActivFunc
    {
        ActivFuncConfig cfg;
        ActivFuncType type;
        real (*ptr)(real, Context&);
        real (*derivPtr)(real, Context&);
    };

    std::vector<ActivFunc> layerActivFunc;

    struct CostFunc
    {
        CostFuncConfig cfg;
        CostFuncType type;
        real (*ptr)(real, real, Context&);
    } costFunc;

    struct OptFunc
    {
        OptFuncConfig cfg;
        OptFuncType type;
        void (*ptr)(Context&);
    } optFunc;

};


} // namespace NeuralNet


#endif // CONFIG_H
