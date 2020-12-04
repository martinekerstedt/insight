#ifndef CONTEXT_H
#define CONTEXT_H

#include "NeuralNet/state.h"
#include "NeuralNet/config.h"


namespace NeuralNet
{


class Context
{

public:
    Context(Config& config, State& state) :
        input(state.input),
        error(state.error),
        avg_input(state.avg_input),
        avg_error(state.avg_error),
        step(state.step),
        initFuncType(config.initFunc.type),
        costFuncType(config.costFunc.type),
        learnRateType(config.learnRateFunc.type),
        optFuncType(config.optFunc.type),        
        initFuncConfig(config.initFunc.cfg),
        costFuncConfig(config.costFunc.cfg),
        learnRateFuncConfig(config.learnRateFunc.cfg),
        optFuncConfig(config.optFunc.cfg),
        sizeVec(config.sizeVec),
        batchSize(config.batchSize),
        printInterval(config.printInterval),
        layers(state.layers),
        learningRate(state.learningRate),
        m_config(config)
    {

    }

    // Current input and error
    VectorView input;
    const Vector& error;

    // Average input and error over a batch
    const Vector& avg_input;
    const Vector& avg_error;

    // Model step
    const unsigned& step;

    // Currently selected functions, read only
    const Config::InitFuncType& initFuncType;
    const Config::ActivFuncType& activFuncType(unsigned layerIdx) { return m_config.layerActivFunc[layerIdx].type; }
//    const std::vector<State::ActivFuncType>& activFuncType2 = m_state.layerActivFunc[].type;
    const Config::CostFuncType& costFuncType;
    const Config::LearningRateFuncType& learnRateType;
    const Config::OptFuncType& optFuncType;

    // Function configs
    const Config::InitFuncConfig& initFuncConfig;
    const Config::ActivFuncConfig& activFuncConfig(unsigned layerIdx) { return m_config.layerActivFunc[layerIdx].cfg; }
    const Config::CostFuncConfig& costFuncConfig;
    const Config::LearningRateFuncConfig& learnRateFuncConfig;
    const Config::OptFuncConfig& optFuncConfig;

    // All user definable functions
    void initializeFunction()
    {
        m_config.initFunc.ptr(*this);
    }

    real activationFunction(real x, unsigned int layerIdx)
    {
        return m_config.layerActivFunc[layerIdx].ptr(x, *this);
    }

    real activationFunctionDerivate(real x, unsigned int layerIdx)
    {
        return m_config.layerActivFunc[layerIdx].derivPtr(x, *this);
    }

    real costFunction(real output, real target)
    {
        return m_config.costFunc.ptr(output, target, *this);
    }

    void optimizeFunction()
    {
        m_config.optFunc.ptr(*this);
    }

    // All user definable function pointers
    auto activationFunction(unsigned int layerIdx)
    {
        return m_config.layerActivFunc[layerIdx].ptr;
    }

    auto activationFunctionDerivate(unsigned int layerIdx)
    {
        return m_config.layerActivFunc[layerIdx].derivPtr;
    }

    // All network configs, read only
    const std::vector<size_t>& sizeVec;
    const unsigned int& batchSize;
    const unsigned int& printInterval;

    // Layers
    std::vector<State::Layer>& layers;

    // Learning rate
    real& learningRate;

private:
    Config& m_config;

};


} // namespace NeuralNet


#endif // CONTEXT_H







