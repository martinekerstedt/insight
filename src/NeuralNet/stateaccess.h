#ifndef STATEACCESS_H
#define STATEACCESS_H

#include "NeuralNet/state.h"


namespace NeuralNet
{


//class StateAccess
//{

//public:
//    StateAccess(State& state) :
//        m_state(state)
//    {

//    }

//    const Vector& input() { return *m_state.input; }
//    const Vector& error() { return m_state.error; }

//    // Currently selected functions, read only
//    State::InitFuncType initFuncType() { return m_state.initFunc.type; }
//    State::ActivFuncType activFuncType(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].type; }
//    State::CostFuncType costFuncType() { return m_state.costFunc.type; }
//    State::OptFuncType optFuncType() { return m_state.optFunc.type; }

//    // Function configs
//    const State::InitFuncConfig& initFuncConfig() { return m_state.initFunc.cfg; }
//    const State::ActivFuncConfig& activFuncConfig(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].cfg; }
//    const State::CostFuncConfig& costFuncConfig() { return m_state.costFunc.cfg; }
//    const State::OptFuncConfig& optFuncConfig() { return m_state.optFunc.cfg; }

//    // All user definable functions
//    void initializeFunction()
//    {
//        m_state.initFunc.ptr(*this);
//    }

//    real activationFunction(real x, unsigned int layerIdx)
//    {
//        return m_state.layerActivFunc[layerIdx].ptr(x, *this);
//    }

//    real activationFunctionDerivate(real x, unsigned int layerIdx)
//    {
//        return m_state.layerActivFunc[layerIdx].derivPtr(x, *this);
//    }

//    auto activationFunction(unsigned int layerIdx)
//    {
//        return m_state.layerActivFunc[layerIdx].ptr;
//    }

//    auto activationFunctionDerivate(unsigned int layerIdx)
//    {
//        return m_state.layerActivFunc[layerIdx].derivPtr;
//    }

//    real costFunction(real output, real target)
//    {
//        return m_state.costFunc.ptr(output, target, *this);
//    }

//    void optimizeFunction(const Vector& input, const Vector& error)
//    {
//        m_state.optFunc.ptr(input, error, *this);
//    }

//    // All network configs, read only
//    const State::Config config() { return m_state.config; }

//    // Network stats, read only
//    // timeSpentTraining
//    // all kinds of time metrics really
//    // nSteps

//    // Layers
//    std::vector<State::Layer>& layers() { return m_state.layers; }

//    // Initialization function declarations
//    friend void init_func_random_normal(StateAccess& net);
//    friend void init_func_random_uniform(StateAccess& net);

//    // Activation function declarations
//    friend Vector activation_func_relu(const Vector& x, StateAccess& net);
//    friend Vector activation_func_sigmoid(const Vector& x, StateAccess& net);
//    friend Vector activation_func_tanh(const Vector& x, StateAccess& net);

//    // Activation function derivative declarations
//    friend Vector activation_func_relu_deriv(const Vector& x, StateAccess& net);
//    friend Vector activation_func_sigmoid_deriv(const Vector& x, StateAccess& net);
//    friend Vector activation_func_tanh_deriv(const Vector& x, StateAccess& net);

//    // Cost function declarations
//    friend Vector cost_func_difference(const Vector& output, const Vector& target, StateAccess& net);
//    friend Vector cost_func_square_difference(const Vector& output, const Vector& target, StateAccess& net);
//    friend Vector cost_func_cross_entropy(const Vector& output, const Vector& target, StateAccess& net);

//    // Optimizer function declarations
//    friend void optimize_func_backprop(const Vector& input, const Vector& error, StateAccess& net);

//private:
//    State& m_state;

//};

class StateAccess
{

public:
    StateAccess(State& state) :
        input(state.input),
        error(state.error),
        avg_input(state.avg_input),
        avg_error(state.avg_error),
        initFuncType(state.initFunc.type),
        costFuncType(state.costFunc.type),
        optFuncType(state.optFunc.type),
        initFuncConfig(state.initFunc.cfg),
        costFuncConfig(state.costFunc.cfg),
        optFuncConfig(state.optFunc.cfg),
        config(state.config),
        layers(state.layers),
        m_state(state)
    {

    }

    // Current input and error
    const Vector* input;
    const Vector& error;

    // Average input and error over a batch
    const Vector& avg_input;
    const Vector& avg_error;

    // Currently selected functions, read only
    const State::InitFuncType& initFuncType;
    const State::ActivFuncType& activFuncType(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].type; }
//    const std::vector<State::ActivFuncType>& activFuncType2 = m_state.layerActivFunc[].type;
    const State::CostFuncType& costFuncType;
    const State::OptFuncType& optFuncType;

    // Function configs
    const State::InitFuncConfig& initFuncConfig;
    const State::ActivFuncConfig& activFuncConfig(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].cfg; }
    const State::CostFuncConfig& costFuncConfig;
    const State::OptFuncConfig& optFuncConfig;

    // All user definable functions
    void initializeFunction()
    {
        m_state.initFunc.ptr(*this);
    }

    real activationFunction(real x, unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].ptr(x, *this);
    }

    real activationFunctionDerivate(real x, unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].derivPtr(x, *this);
    }

    real costFunction(real output, real target)
    {
        return m_state.costFunc.ptr(output, target, *this);
    }

    void optimizeFunction()
    {
        m_state.optFunc.ptr(*this);
    }

    // All user definable function pointers
    auto activationFunction(unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].ptr;
    }

    auto activationFunctionDerivate(unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].derivPtr;
    }

    // All network configs, read only
    const State::Config& config;

    // Layers
    std::vector<State::Layer>& layers;

private:
    State& m_state;

};


} // namespace NeuralNet


#endif // STATEACCESS_H







