#include <NeuralNet/functions.h>
#include <cmath>
#include <random>

// Initialization functions
void init_func_random_normal(NeuralNet::StateAccess& net)
{
    InitializationFunction::RANDOM_NORMAL cfg = net.m_state.initFunc.cfg.random;

    // Random generator
    std::random_device rd{};
    std::mt19937 gen{rd()};

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
Vector activation_func_relu(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    Vector res(x.size(), 0.0);

    for (unsigned i = 0; i < x.size(); ++i) {

        real val = x(i);

        if (val >= 0.0) {
            res(i) = val;
        } else {
            if (val < -15.0) {
                res(i) = -0.2;
            } else {
                res(i) = 0.2*(std::pow(e, val) - 1.0);
            }
        }
    }

    return res;
}

Vector activation_func_sigmoid(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    Vector res(x.size(), 0.0);

    for (unsigned i = 0; i < x.size(); ++i) {

        real val = x(i);

        if (val > 15.0) {
            res(i) = 1.0;
        } else if (val < -15.0) {
            res(i) = 0.0;
        } else {
            res(i) = 1.0 / (1.0 + std::pow(e, -val));
        }

    }

    return res;
}

Vector activation_func_tanh(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    Vector res(x.size(), 0.0);

    for (unsigned i = 0; i < x.size(); ++i) {

        real val = x(i);

        if (val > 15.0) {
            res(i) = 1.0;
        } else if (val < -15.0) {
            res(i) = -1.0;
        } else {
            res(i) = std::tanh(val);
        }

    }

    return res;
}

// Activation function derivatives
Vector activation_func_relu_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    Vector res(x.size(), 1);

    for (unsigned i = 0; i < x.size(); ++i) {

        real val = x(i);

        if (val >= 0) {
            res(i) = 1.0;
        } else {
            if (val < -15.0) {
                res(i) = 0;
            } else {
                res(i) = 0.2*std::pow(e, val);
            }
        }

    }

    return res;
}

Vector activation_func_sigmoid_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    static const real e = static_cast<real>(std::exp(1.0));

    Vector res(x.size(), 1);

    for (unsigned i = 0; i < x.size(); ++i) {

        real val = x(i);

        if (val > 15) {
            res(i) = 0;
        } else if (val < -15) {
            res(i) = 0;
        } else {
            res(i) = std::pow(e, -val) / std::pow(1 + std::pow(e, -val), 2);
        }
    }

    return res;
}

Vector activation_func_tanh_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
{
    (void)layerIdx;
    (void)net;

    Vector res(x.size(), 1);

    for (unsigned i = 0; i < x.size(); ++i) {
        res(i) = 2 / (std::cosh(2*x(i)) + 1);
    }

    return res;
}

// Cost functions
Vector cost_func_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
{
    (void)net;

    return output - target;
}

Vector cost_func_square_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
{
    (void)net;

    Vector diff = output - target;

    for (unsigned i = 0; i < diff.size(); ++i) {
        real& val = diff(i);

        if (val >= 0.0) {
            val = std::pow(val, 2);
        } else {
            val = -std::pow(val, 2);
        }
    }

    return diff;
}

Vector cost_func_cross_entropy(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
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

    Vector res(output.size(), 1);

    for (unsigned i = 0; i < output.size(); ++i) {
        if (target(i) == 1.0) {
            res(i) = -std::log(output(i));
        } else {
            res(i) = -std::log(1 - output(i));
        }
    }

    return res;
}

// Optimize functions
void optimize_func_backprop(const Vector& input, const Vector& error, NeuralNet::StateAccess& net)
{
    std::vector<NeuralNet::Layer>& layers = net.m_state.layers;

    // Backprop output layer
    layers.back().gradient = error*net.m_state.config.learningRate;

    // Loop backwards
    for (int i = (layers.size() - 2); i >= 0; --i) {

        layers[i].gradient = layers[i + 1].weights.transpose()
                * (net.activationFunctionDerivate(i, layers[i + 1].weightedSum).multiplyElemWise(layers[i + 1].gradient));
    }

    // Loop backwards
    for (int i = (layers.size() - 1); i > 0; --i) {

        layers[i].weights -= net.activationFunctionDerivate(i, layers[i].weightedSum).multiplyElemWise(layers[i].gradient)
                * layers[i - 1].output.transpose();

    }

    layers[0].weights -= net.activationFunctionDerivate(0, layers[0].weightedSum).multiplyElemWise(layers[0].gradient)
            * input.transpose();
}
