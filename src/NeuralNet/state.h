#ifndef STATE_H
#define STATE_H

#include <NeuralNet/matrix.h>
#include <NeuralNet/vector.h>
#include <NeuralNet/vectorview.h>

namespace NeuralNet
{


class StateAccess;

struct State
{
    // Configuration
    struct Config
    {
        std::vector<size_t> sizeVec;
        unsigned int batchSize;
        unsigned int nEpochs;
        unsigned int printInterval;
        bool softMax;
    } config;

    // Precepton layer
    struct Layer
    {
        Layer(size_t prevLayerSize, size_t size) :
            output(size, 0.0),
            bias(size, 0.0),
            gradient(size, 0.0),
            weightedSum(size, 0.0),
            weights(size, prevLayerSize, 0.0),
            m_size(size)
        {

        }

        Vector output;
        Vector bias;
        Vector gradient;
        Vector weightedSum;
        Matrix weights;

        size_t size()
        {
            return m_size;
        }

    private:
        size_t m_size;

    };

    std::vector<Layer> layers;

    // Input
//    const Vector* input; // Should perhaps be a view type
    VectorView input;
    Vector avg_input;

    // Target
//    const Vector* target;
    VectorView target;

    // Error
    Vector error;
    Vector avg_error;

    // Step
    unsigned long step;

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
        void (*ptr)(StateAccess&);
    } initFunc;

    struct ActivFunc
    {
        ActivFuncConfig cfg;
        ActivFuncType type;
        real (*ptr)(real, StateAccess&);
        real (*derivPtr)(real, StateAccess&);
    };

    std::vector<ActivFunc> layerActivFunc;

    struct CostFunc
    {
        CostFuncConfig cfg;
        CostFuncType type;
        real (*ptr)(real, real, StateAccess&);
    } costFunc;

    struct OptFunc
    {
        OptFuncConfig cfg;
        OptFuncType type;
        void (*ptr)(StateAccess&);
    } optFunc;

};


} // namespace NeuralNet


#endif // STATE_H































































