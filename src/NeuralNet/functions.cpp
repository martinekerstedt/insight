#include <NeuralNet/functions.h>
#include <cmath>
#include <random>


namespace NeuralNet
{


// Initialization functions
void init_func_random_normal(NeuralNet::StateAccess& net)
{
//    InitializationFunction::RANDOM_NORMAL cfg = net.m_state.initFunc.cfg.random;
    InitializationFunction::RANDOM_NORMAL cfg = net.initFuncConfig.random;

    std::mt19937 gen;

    if (cfg.seed == 0) {
        // Random generator
        std::random_device rd{};
        gen.seed(rd());
    } else {
        gen.seed(cfg.seed);
    }

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<real> d{cfg.mean, cfg.stddev};       

    // Get layers
    std::vector<State::Layer>& layers = net.layers;

    // Loop layers
    for (size_t i = 0; i < layers.size(); ++i) {
        // Loop neurons
        for (size_t j = 0; j < layers[i].size(); ++j) {
            // Set bias
//            net->layers()[i][j].bias = 0; // default is zero

            // Loop weights
            for (size_t k = 0; k < layers[i].weights.cols(); ++k) {
                layers[i].weights(j, k) = d(gen);
            }
        }
    }
}

void init_func_random_uniform(NeuralNet::StateAccess& net)
{
//    InitializationFunction::RANDOM_UNIFORM cfg = net.m_state.initFunc.cfg.uniform;
    InitializationFunction::RANDOM_UNIFORM cfg = net.initFuncConfig.uniform;

    std::mt19937 gen;

    if (cfg.seed == 0) {
        // Random generator
        std::random_device rd{};
        gen.seed(rd());
    } else {
        gen.seed(cfg.seed);
    }


    // Get layers
    std::vector<State::Layer>& layers = net.layers;


    // Input layer
    real max = 1.0/std::sqrt(net.config.sizeVec[0]*layers[0].size());
    real min = -1.0/std::sqrt(net.config.sizeVec[0]*layers[0].size());

    std::uniform_real_distribution<> d(min, max);

    // Loop neurons
    for (size_t j = 0; j < layers[0].size(); ++j) {

        // Loop weights
        for (size_t k = 0; k < layers[0].weights.cols(); ++k) {
            layers[0].weights(j, k) = d(gen);
        }
    }


    // Loop layers
    for (size_t i = 1; i < layers.size(); ++i) {

        real max = 1.0/std::sqrt(layers[i - 1].size()*layers[i].size());
        real min = -1.0/std::sqrt(layers[i - 1].size()*layers[i].size());

        std::uniform_real_distribution<> d2(min, max);

        // Loop neurons
        for (size_t j = 0; j < layers[i].size(); ++j) {

            // Loop weights
            for (size_t k = 0; k < layers[i].weights.cols(); ++k) {
                layers[i].weights(j, k) = d2(gen);
            }
        }
    }
}

// Activation functions
real activation_func_relu(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    if (x >= 0.0) {
        return x;
    } else {
        if (x < -15.0) {
            return -0.2;
        } else {
            return 0.2*(std::pow(e, x) - 1.0);
        }
    }
}

real activation_func_sigmoid(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    if (x > 15.0) {
        return 1.0;
    } else if (x < -15.0) {
        return 0.0;
    } else {
        return 1.0 / (1.0 + std::pow(e, -x));
    }
}

real activation_func_tanh(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    if (x > 15.0) {
        return 1.0;
    } else if (x < -15.0) {
        return -1.0;
    } else {
        return std::tanh(x);
    }
}

// Activation function derivatives
real activation_func_relu_deriv(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    if (x >= 0) {
        return 1.0;
    } else {
        if (x < -15.0) {
            return 0;
        } else {
            return 0.2*std::pow(e, x);
        }
    }
}

real activation_func_sigmoid_deriv(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    if (x > 15) {
        return 0;
    } else if (x < -15) {
        return 0;
    } else {
        return std::pow(e, -x) / std::pow(1 + std::pow(e, -x), 2);
    }
}

real activation_func_tanh_deriv(real x, NeuralNet::StateAccess& net)
{
    (void)net;

    return 2 / (std::cosh(2*x) + 1);
}

// Cost functions
real cost_func_difference(real output, real target, NeuralNet::StateAccess& net)
{
    (void)net;

    return output - target;
}

real cost_func_square_difference(real output, real target, NeuralNet::StateAccess& net)
{
    (void)net;

    real diff = output - target;

    if (diff >= 0.0) {
        return std::pow(diff, 2);
    } else {
        return -std::pow(diff, 2);
    }
}

real cost_func_cross_entropy(real output, real target, NeuralNet::StateAccess& net)
{
    (void)net;

    // Assumes that target is either 1 or 0
    // And that output is between 0 and 1
//    if ((output > 1.0) || (output < 0.0)) {
//        THROW_ERROR("Invalid value of output: "
//                    << output
//                    << ", Expected: Between 0 and 1");
//    }

//    if ((target > 1.0) || (target < 0.0)) {
//        THROW_ERROR("Invalid value of target: "
//                    << output
//                    << ", Expected: Between 0 and 1");
//    }

    if (target == 1.0) {
        return -std::log(output);
    } else {
        return -std::log(1 - output);
    }
}

// Optimize functions
void optimize_func_backprop(const Vector& input, const Vector& error, NeuralNet::StateAccess& net)
{
//    OptimizeFunction::BACKPROP& cfg = net.m_state.optFunc.cfg.backprop;
//    std::vector<State::Layer>& layers = net.m_state.layers;
//    auto afd = net.m_state.layerActivFunc[0].derivPtr;
    const OptimizeFunction::BACKPROP& cfg = net.optFuncConfig.backprop;
    std::vector<State::Layer>& layers = net.layers;
    auto afd = net.activationFunctionDerivate(0);

    // Backprop output layer
    layers.back().gradient = error*cfg.learningRate;

    // Loop backwards
    for (int i = (layers.size() - 2); i >= 0; --i) {

        layers[i].gradient = layers[i + 1].weights.trans()
                * (Matrix::apply(layers[i + 1].weightedSum, afd, net) ** layers[i + 1].gradient);
    }

    // Loop backwards
    for (int i = (layers.size() - 1); i > 0; --i) {

        layers[i].weights -= (Matrix::apply(layers[i].weightedSum, afd, net) ** layers[i].gradient)
                * layers[i - 1].output.trans();
    }

    layers[0].weights -= (Matrix::apply(layers[0].weightedSum, afd, net) ** layers[0].gradient)
            * input.trans();
}


} // namespace NeuralNet



































