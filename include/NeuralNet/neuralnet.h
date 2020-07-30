#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Common/types.h>

//
// TODO
//
// DataSet type, should make things simpler and more easy to understand
//
// Parameters
//   Batch size
//   Number of epochs
//   Toggle soft max on the output
//   Learning rate, actually a parameter to the cost functions, scale cost function
//   Sampeling/printing rate, different for training and testing
//   Design to make it very simple to write custom cost/activ/opt/init-functions
//     Cost function
//     Activation function, per layer
//     Optimizer funciton/Backprop function
//     Initialization function
//
// Functions
//   Propergate
//   Start training, will return when done
//   Stop training
//   Get weights and biases, return a string or vectors
//   Load weights and biases
//   Get current output
//   Get current error vector
//   Get full access to internal state, the "insight" in Insight
//
// Data type to replace real in most places, probably only need -+100 or +-1000 range
//   (float is 32bit, +-1000 would be 12bit)
//

struct Neuron
{
    Neuron(size_t size) : size(size) {}
    const size_t size;
    real_vec weights;
    real bias = 0;
    real output = 0;
    real gradient = 0;
    real weightedSum = 0;
};

struct Layer
{
    enum class Type
    {
        INPUT,
        HIDDEN,
        OUTPUT,
        INPUT_OUTPUT
    };

    enum class Activation
    {
        RELU,
        SIGMOID,
        TANH
    };

    Layer(size_t size,
          Layer::Type type,
          Layer::Activation activation = Activation::RELU,
          bool softMax = false) :
        size(size),
        type(type),
        activation(activation),
        softMax(softMax)
    {

    }

    Neuron& operator [](int i)
    {
        return neurons[i];
    }

    Neuron operator [](int i) const
    {
        return neurons[i];
    }

    const size_t size;
    const Type type;
    Activation activation;
    bool softMax;

    std::vector<Neuron> neurons;

};

class NeuralNet
{
public:
    enum class CostFunction
    {
        DIFFERENCE,
        CROSS_ENTROPY,
        SQUARE_DIFFERENCE
    };

    NeuralNet(std::vector<size_t> size);

    void propergate(real_vec input);
    void calcError(real_vec target);
    real train(real_matrix input, real_matrix target);
    
    std::vector<Layer> layers;
    std::vector<size_t> size_vec();

    unsigned int batchSize;
    unsigned int nEpochs;
    unsigned int printInterval;
    bool softMaxOutput;
    CostFunction costFunction;
    real learningRate;
    real momentum;

    void setHiddenActivation(Layer::Activation activation);
    void setOutputActivation(Layer::Activation activation);

private:
    void backpropergate(real_vec input, real_vec error);
    void printState(real_vec input, real_vec target, size_t batchIdx);

    real costFunc(real output, real target);

    real activationFunction(Layer::Activation func, real x);
    real activationFunctionDerivative(Layer::Activation func, real x);

    void softMax(real_vec& vec);
    void softMax(Layer& vec);

    void updateLearningRate(real cost);

    std::vector<size_t> m_size_vec;
    real_vec m_error;

    const real m_eulerConstant;

};

#endif // NEURAL_NET_H
