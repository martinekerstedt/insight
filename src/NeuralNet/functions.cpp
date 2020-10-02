#include <NeuralNet/functions.h>
#include <cmath>
#include <random>

// Initialization functions
void init_func_random_normal(NeuralNet::StateAccess& net)
{
    InitializationFunction::RANDOM_NORMAL cfg = net.m_state.initFunc.cfg.random;

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
    std::vector<NeuralNet::Layer>& layers = net.layers();

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
    OptimizeFunction::BACKPROP& cfg = net.m_state.optFunc.cfg.backprop;
    std::vector<NeuralNet::Layer>& layers = net.m_state.layers;

    // Backprop output layer
    layers.back().gradient = error*cfg.learningRate;

    // Loop backwards
    for (int i = (layers.size() - 2); i >= 0; --i) {

        layers[i].gradient = layers[i + 1].weights.trans()
                * Matrix::mulEWise(Matrix::apply(layers[i + 1].weightedSum, activation_func_sigmoid_deriv, net), layers[i + 1].gradient);
    }

    // Loop backwards
    for (int i = (layers.size() - 1); i > 0; --i) {

        layers[i].weights -=Matrix::mulEWise(Matrix::apply(layers[i].weightedSum, activation_func_sigmoid_deriv, net), layers[i].gradient)
                * layers[i - 1].output.trans();
    }

    layers[0].weights -= Matrix::mulEWise(Matrix::apply(layers[0].weightedSum, activation_func_sigmoid_deriv, net), layers[0].gradient)
            * input.trans();
}







































