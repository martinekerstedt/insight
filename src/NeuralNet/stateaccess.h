#ifndef STATEACCESS_H
#define STATEACCESS_H

#include "NeuralNet/state.h"


namespace NeuralNet
{


class StateAccess
{

public:
    StateAccess(State& state) :
        m_state(state)
    {

    }

    // Currently selected functions, read only
    State::InitFuncType initFuncType() { return m_state.initFunc.type; }
    State::ActivFuncType activFuncType(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].type; }
    State::CostFuncType costFuncType() { return m_state.costFunc.type; }
    State::OptFuncType optFuncType() { return m_state.optFunc.type; }


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

    auto activationFunction(unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].ptr;
    }

    auto activationFunctionDerivate(unsigned int layerIdx)
    {
        return m_state.layerActivFunc[layerIdx].derivPtr;
    }

    real costFunction(real output, real target)
    {
        return m_state.costFunc.ptr(output, target, *this);
    }

    void optimizeFunction(const Vector& input, const Vector& error)
    {
        m_state.optFunc.ptr(input, error, *this);
    }

    // All network configs, read only
    const State::Config config() { return m_state.config; }

    // Network stats, read only
    // timeSpentTraining
    // all kinds of time metrics really
    // nSteps

    // Layers
    std::vector<State::Layer>& layers() { return m_state.layers; }

    // Initialization function declarations
    friend void init_func_random_normal(StateAccess& net);
    friend void init_func_random_uniform(StateAccess& net);

    // Activation function declarations
    friend Vector activation_func_relu(const Vector& x, StateAccess& net);
    friend Vector activation_func_sigmoid(const Vector& x, StateAccess& net);
    friend Vector activation_func_tanh(const Vector& x, StateAccess& net);

    // Activation function derivative declarations
    friend Vector activation_func_relu_deriv(const Vector& x, StateAccess& net);
    friend Vector activation_func_sigmoid_deriv(const Vector& x, StateAccess& net);
    friend Vector activation_func_tanh_deriv(const Vector& x, StateAccess& net);

    // Cost function declarations
    friend Vector cost_func_difference(const Vector& output, const Vector& target, StateAccess& net);
    friend Vector cost_func_square_difference(const Vector& output, const Vector& target, StateAccess& net);
    friend Vector cost_func_cross_entropy(const Vector& output, const Vector& target, StateAccess& net);

    // Optimizer function declarations
    friend void optimize_func_backprop(const Vector& input, const Vector& error, StateAccess& net);

private:
    State& m_state;

};


} // namespace NeuralNet


#endif // STATEACCESS_H







