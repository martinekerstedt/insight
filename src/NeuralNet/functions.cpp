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
//Vector activation_func_relu(const Vector& x, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    static const real e = static_cast<real>(std::exp(1.0));

//    Vector res(x.size(), 0.0);

//    for (unsigned i = 0; i < x.size(); ++i) {

//        real val = x(i);

//        if (val >= 0.0) {
//            res(i) = val;
//        } else {
//            if (val < -15.0) {
//                res(i) = -0.2;
//            } else {
//                res(i) = 0.2*(std::pow(e, val) - 1.0);
//            }
//        }
//    }

//    return res;
//}

//Vector activation_func_sigmoid(const Vector& x, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    static const real e = static_cast<real>(std::exp(1.0));

//    Vector res(x.size(), 0.0);

//    for (unsigned i = 0; i < x.size(); ++i) {

//        real val = x(i);

//        if (val > 15.0) {
//            res(i) = 1.0;
//        } else if (val < -15.0) {
//            res(i) = 0.0;
//        } else {
//            res(i) = 1.0 / (1.0 + std::pow(e, -val));
//        }

//    }

//    return res;
//}

//Vector activation_func_tanh(const Vector& x, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    Vector res(x.size(), 0.0);

//    for (unsigned i = 0; i < x.size(); ++i) {

//        real val = x(i);

//        if (val > 15.0) {
//            res(i) = 1.0;
//        } else if (val < -15.0) {
//            res(i) = -1.0;
//        } else {
//            res(i) = std::tanh(val);
//        }

//    }

//    return res;
//}

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
//Vector activation_func_relu_deriv(const Vector& x, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    static const real e = static_cast<real>(std::exp(1.0));

//    Vector res(x.size(), 1);

//    for (unsigned i = 0; i < x.size(); ++i) {

//        real val = x(i);

//        if (val >= 0) {
//            res(i) = 1.0;
//        } else {
//            if (val < -15.0) {
//                res(i) = 0;
//            } else {
//                res(i) = 0.2*std::pow(e, val);
//            }
//        }

//    }

//    return res;
//}

//Vector activation_func_sigmoid_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net)
//{
//    (void)layerIdx;
//    (void)net;

//    static const real e = static_cast<real>(std::exp(1.0));

//    Vector res(x.size(), 1);

//    for (unsigned i = 0; i < x.size(); ++i) {

//        real val = x(i);

//        if (val > 15) {
//            res(i) = 0;
//        } else if (val < -15) {
//            res(i) = 0;
//        } else {
//            res(i) = std::pow(e, -val) / std::pow(1 + std::pow(e, -val), 2);
//        }
//    }

//    return res;
//}

//Vector activation_func_tanh_deriv(const Vector& x, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    Vector res(x.size(), 1);

//    for (unsigned i = 0; i < x.size(); ++i) {
//        res(i) = 2 / (std::cosh(2*x(i)) + 1);
//    }

//    return res;
//}

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
//Vector cost_func_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    return output - target;
//}

//Vector cost_func_square_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    Vector diff = output - target;

//    for (unsigned i = 0; i < diff.size(); ++i) {
//        real& val = diff(i);

//        if (val >= 0.0) {
//            val = std::pow(val, 2);
//        } else {
//            val = -std::pow(val, 2);
//        }
//    }

//    return diff;
//}

//Vector cost_func_cross_entropy(const Vector& output, const Vector& target, NeuralNet::StateAccess& net)
//{
//    (void)net;

//    // Assumes that target is either 1 or 0
//    // And that output is between 0 and 1
////    if ((output > 1.0) || (output < 0.0)) {
////        THROW_ERROR("Invalid value of output: "
////                    << output
////                    << ", Expected: Between 0 and 1");
////    }

////    if ((target > 1.0) || (target < 0.0)) {
////        THROW_ERROR("Invalid value of target: "
////                    << output
////                    << ", Expected: Between 0 and 1");
////    }

//    Vector res(output.size(), 1);

//    for (unsigned i = 0; i < output.size(); ++i) {
//        if (target(i) == 1.0) {
//            res(i) = -std::log(output(i));
//        } else {
//            res(i) = -std::log(1 - output(i));
//        }
//    }

//    return res;
//}

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
//        layers[i].gradient = layers[i + 1].weights.trans()
//                * (net.activationFunctionDerivate(i, layers[i + 1].weightedSum).mulEWise(layers[i + 1].gradient));

        layers[i].gradient = layers[i + 1].weights.trans() * Matrix::mulEWise(Matrix::apply(activation_func_sigmoid_deriv, layers[i + 1].weightedSum, net), layers[i + 1].gradient);
//        layers[i].gradient = layers[i + 1].weights.trans() * Matrix::mulEWise(net.activationFunctionDerivate(i, layers[i + 1].weightedSum), layers[i + 1].gradient);
    }

    // Loop backwards
    for (int i = (layers.size() - 1); i > 0; --i) {

//        layers[i].weights -= net.activationFunctionDerivate(i, layers[i].weightedSum).mulEWise(layers[i].gradient)
//                * layers[i - 1].output.trans();

        layers[i].weights -= Matrix::mulEWise(Matrix::apply(activation_func_sigmoid_deriv, layers[i].weightedSum, net), layers[i].gradient) * layers[i - 1].output.trans();
    }

//    layers[0].weights -= net.activationFunctionDerivate(0, layers[0].weightedSum).mulEWise(layers[0].gradient)
//            * input.trans();

    layers[0].weights -= Matrix::mulEWise(Matrix::apply(activation_func_sigmoid_deriv, layers[0].weightedSum, net), layers[0].gradient) * input.trans();
}







































