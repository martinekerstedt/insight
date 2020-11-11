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
        initFuncType(config.initFunc.type),
        costFuncType(config.costFunc.type),
        optFuncType(config.optFunc.type),
        initFuncConfig(config.initFunc.cfg),
        costFuncConfig(config.costFunc.cfg),
        optFuncConfig(config.optFunc.cfg),
        sizeVec(config.sizeVec),
        batchSize(config.batchSize),
        printInterval(config.printInterval),
        layers(state.layers),
        m_config(config),
        m_state(state)
    {

    }

    // Current input and error
    VectorView input;
    const Vector& error;

    // Average input and error over a batch
    const Vector& avg_input;
    const Vector& avg_error;

    // Currently selected functions, read only
    const Config::InitFuncType& initFuncType;
    const Config::ActivFuncType& activFuncType(unsigned layerIdx) { return m_config.layerActivFunc[layerIdx].type; }
//    const std::vector<State::ActivFuncType>& activFuncType2 = m_state.layerActivFunc[].type;
    const Config::CostFuncType& costFuncType;
    const Config::OptFuncType& optFuncType;

    // Function configs
    const Config::InitFuncConfig& initFuncConfig;
    const Config::ActivFuncConfig& activFuncConfig(unsigned layerIdx) { return m_config.layerActivFunc[layerIdx].cfg; }
    const Config::CostFuncConfig& costFuncConfig;
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

private:
    Config& m_config;
    State& m_state;

};


} // namespace NeuralNet


#endif // CONTEXT_H







