#include <NeuralNet/neuralnet.h>
#include <cmath>
#include <random>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <iomanip>

// Could perhaps declare these firend functions of StateAccess
// Then there is no need for the configs to be public in StateAccess

// Initialization function declarations
void init_func_random_normal(NeuralNet::StateAccess& net);

// Activation function declarations
Vector activation_func_relu(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);
Vector activation_func_sigmoid(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);
Vector activation_func_tanh(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);

// Activation function derivative declarations
Vector activation_func_relu_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);
Vector activation_func_sigmoid_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);
Vector activation_func_tanh_deriv(const Vector& x, unsigned int layerIdx, NeuralNet::StateAccess& net);

// Cost function declarations
Vector cost_func_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);
Vector cost_func_square_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);
Vector cost_func_cross_entropy(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);

// Optimizer function declarations
void optimize_func_backprop(const Vector& input, const Vector& error, NeuralNet::StateAccess& net);

NeuralNet::NeuralNet() :
    NeuralNet({1, 2, 1})
{

}

NeuralNet::NeuralNet(const std::initializer_list<size_t>& list) :
    NeuralNet(std::vector<size_t>(list))
{

}

NeuralNet::NeuralNet(const std::vector<size_t>& sizeVec) :
    m_stateAccess(m_state)
{
    if (sizeVec.size() < 2) {
        THROW_ERROR("Invalid net size: "
                    << sizeVec.size()
                    << ", Expected: >=2");
    }

    for (size_t i = 0; i < sizeVec.size(); ++i) {
        if (sizeVec[i] < 1) {
            THROW_ERROR("Invalid layer size: "
                        << sizeVec.size()
                        << ", Expected: >=1");
        }
    }

    m_state.config.sizeVec = sizeVec;
    m_state.config.batchSize = 1;
    m_state.config.nEpochs = 1;
    m_state.config.printInterval = 1;
    m_state.config.softMax = false;
    m_state.config.learningRate = 1.0;


    // Default initialization function
    InitializationFunction::RANDOM_NORMAL rnd_cfg;
    setInitializationFunction(rnd_cfg);


    // Default cost function
    CostFunction::SQUARE_DIFFERENCE sq_diff_cfg;
    setCostFunction(sq_diff_cfg);


    // Default optimize function
    OptimizeFunction::BACKPROP backprop_cfg;
    setOptimizeFunction(backprop_cfg);


    // Create network
    for (size_t i = 0; i < (sizeVec.size() - 1); ++i) {

        // Add layer
        m_state.layers.push_back(Layer(sizeVec[i], sizeVec[i + 1]));
        State::ActivFunc activFunc;
        activFunc.type = State::ActivFuncType::SIGMOID;
        activFunc.ptr = activation_func_sigmoid;
        activFunc.derivPtr = activation_func_sigmoid_deriv;
        ActivationFunction::SIGMOID sig_cfg;
        activFunc.cfg.sigmoid = sig_cfg;
        m_state.layerActivFunc.push_back(activFunc);
    }


    // Init weights and biases
    m_state.initFunc.ptr(m_stateAccess);
}

NeuralNet::State::Config &NeuralNet::config()
{
    return m_state.config;
}

void NeuralNet::propergate(const Vector& input)
{
    if (input.size() != m_state.config.sizeVec[0]) {
        THROW_ERROR("Invalid input size: "
                    << input.size()
                    << ", Expected: "
                    << m_state.config.sizeVec[0]);
    }

    // When not training, dont need to cache weightedSum
    // m_layers[i].output = activFunc(matrixAdd(matrixMulti(input, m_layers[i].weights), m_layers[i].bias));

    // Propergate input layer
    m_state.layers[0].weightedSum = (m_state.layers[0].weights * input) + m_state.layers[0].bias;
    m_state.layers[0].output = m_state.layerActivFunc[0].ptr(m_state.layers[0].weightedSum, 0, m_stateAccess);

    for (unsigned i = 1; i < m_state.layers.size(); ++i) {
        // Propergate hidden and output layers
        m_state.layers[i].weightedSum = (m_state.layers[i].weights * m_state.layers[i - 1].output) + m_state.layers[i].bias;
        m_state.layers[i].output = m_state.layerActivFunc[i].ptr(m_state.layers[i].weightedSum, i, m_stateAccess);
    }

    if (m_state.config.softMax) {
        softMax(m_state.layers.back().output);
    }
}

void NeuralNet::train(const Matrix& input, const Matrix& target)
{
    // Check sizes
    if (input.rows() != target.rows()) {
        THROW_ERROR("Input and Target matricies must have equal number of rows.\n"
                    << input.rows()
                    << " != "
                    << target.rows());
    }

    if (input.cols() != m_state.config.sizeVec.front()) {
        THROW_ERROR("Number of cols in input matrix must equal number of input neurons.\n"
                    << input.cols()
                    << " != "
                    << m_state.config.sizeVec.front());
    }

    if (target.cols() != m_state.config.sizeVec.back()) {
        THROW_ERROR("Number of cols in target matrix must equal number of output neurons.\n"
                    << input.cols()
                    << " != "
                    << m_state.config.sizeVec.front());
    }

    // Number of training samples
    size_t nSamples = input.rows();

    // Number of epochs
    for (size_t epoch = 0; epoch < m_state.config.nEpochs; ++epoch) {


        // Change print interval at last epoch
        if (epoch == (m_state.config.nEpochs - 1)) {
            m_state.config.printInterval = 500;
        }


        // Loop training set
        size_t inputIdx = 0;
        while (inputIdx < nSamples) {


            // Training batch
            size_t batchMax = std::min(inputIdx + m_state.config.batchSize, nSamples);

            Vector error(m_state.layers.back().size(), 1);

            Vector avg_error(m_state.layers.back().size(), 0.0);

            Vector avg_input(m_state.config.sizeVec.front(), 0.0);

            for (size_t batchIdx = inputIdx; batchIdx < batchMax; ++batchIdx) {

                // Propergate input vector
                propergate(input.row(batchIdx));

                // Calc error vector, average over batch
                error = m_state.costFunc.ptr(m_state.layers.back().output, target.row(batchIdx), m_stateAccess);

                // Save error to get average
                // Note: avg_error.size() == output.size() == target.size()
                avg_error += error;

                // Average input
                avg_input += input.row(batchIdx);

                // Print
                if ((batchIdx % m_state.config.printInterval) == 0) {
                    printState(input.row(batchIdx), target.row(batchIdx), error, batchIdx);
                }
            }


            // Divide to get avgerages
            size_t nSamples = batchMax - inputIdx;
            avg_error /= nSamples;
            avg_input /= nSamples;


            // Optimize on avg_error
            m_state.optFunc.ptr(avg_input, avg_error, m_stateAccess);


            // Update index
            inputIdx += m_state.config.batchSize;
        }
    }
}

void NeuralNet::softMax(Vector& output)
{
    real sum = 0;

    for (size_t i = 0; i < output.size(); ++i) {
        output(i) = std::exp(output(i));
        sum += output(i);
    }

    for (size_t i = 0; i < output.size(); ++i) {
        output(i) /= sum;
    }
}

const Vector& NeuralNet::output()
{
    return m_state.layers.back().output;
}

void NeuralNet::save(std::string dir)
{
    // Need to save m_sizeVec, weights and biases
    // Maybe also all other config

    std::stringstream netStr;

    // Network size
    netStr << "size:\n";

    for (unsigned i = 0; i < m_state.config.sizeVec.size(); ++i) {
        netStr << m_state.config.sizeVec[i] << "\n";
    }

    // Loop layers
    for (unsigned i = 0; i < m_state.layers.size(); ++i) {

        netStr << "layer " << i << ":\n";

        // Weights
        netStr << "weights:\n";

        for (unsigned j = 0; j < m_state.layers[i].weights.size(); ++j) {
            netStr << m_state.layers[i].weights(j) << "\n";
        }


        // Bias
        netStr << "bias:\n";

        for (unsigned j = 0; m_state.layers[i].bias.size(); ++j) {
            netStr << m_state.layers[i].bias(j) << "\n";
        }


    }
}

void NeuralNet::printState(Vector input, Vector target, Vector error, size_t batchIdx)
{
    // Create stream
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);



    // Index
    ss << "idx: [ " << batchIdx << " ]\n";



    // Learning rate
    ss << "lrt: [ " << std::setprecision(4) << m_state.config.learningRate << std::setprecision(2) << " ]\n";



    // Input
//    ss << "inp: [ " << input[0];

//    for (size_t i = 1; i < input.size(); ++i) {
//        ss << ",  " << input[i];
//    }

//    ss<< " ]\n";



    // Output
    real output = m_state.layers.back().output(0);

    if (output < 0) {
        ss << "out: [" << output;
    } else {
        ss << "out: [ " << output;
    }

    for (size_t i = 1; i < m_state.layers.back().size(); ++i) {
        output = m_state.layers.back().output(i);

        if (output < 0) {
            ss << ", " << output;
        } else {
            ss << ",  " << output;
        }
    }

    ss << " ]\n";



    // Target
    ss << "tar: [ " << target(0);

    for (size_t i = 1; i < target.size(); ++i) {
        ss << ",  " << target(i);
    }

    ss << " ]\n";



    // Error
    real err = error(0);

    if (err < 0) {
        ss << "err: [" << err;
    } else {
        ss << "err: [ " << err;
    }

    for (size_t i = 1; i < error.size(); ++i) {
        err = error(i);

        if (err < 0) {
            ss << ", " << err;
        } else {
            ss << ",  " << err;
        }
    }

    ss << " ]\n";



    // Cost
    real cost = 0;
    for (size_t i = 0; i < error.size(); ++i) {
        cost += std::pow(error(i), 2);
    }

    if (cost < 0) {
        ss << "cst: [" << cost << " ]\n";
    } else {
        ss << "cst: [ " << cost << " ]\n";
    }



    // Print stream
    std::cout << ss.str() << std::endl;
}

// Initialization functions
void NeuralNet::setInitializationFunction(InitializationFunction::ALL_ZERO init_func)
{
    m_state.initFunc.cfg.all_zero = init_func;
    m_state.initFunc.type = State::InitFuncType::ALL_ZERO;
    m_state.initFunc.ptr = init_func_random_normal;
}

void NeuralNet::setInitializationFunction(InitializationFunction::RANDOM_NORMAL init_func)
{
    m_state.initFunc.cfg.random = init_func;
    m_state.initFunc.type = State::InitFuncType::RANDOM;
    m_state.initFunc.ptr = init_func_random_normal;
}

void NeuralNet::setInitializationFunction(void (*initFunc)(StateAccess&))
{
    m_state.initFunc.type = State::InitFuncType::CUSTOM;
    m_state.initFunc.ptr = initFunc;
}

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
Vector NeuralNet::activationFunction(unsigned int layerIdx, const Vector& x)
{
    return m_state.layerActivFunc[layerIdx].ptr(x, layerIdx, m_stateAccess);
}

Vector NeuralNet::activationFunctionDerivate(unsigned int layerIdx, const Vector& x)
{
    return m_state.layerActivFunc[layerIdx].derivPtr(x, layerIdx, m_stateAccess);
}

void NeuralNet::setActivationFunction(unsigned int layerIdx, ActivationFunction::RELU activ_func)
{
    m_state.layerActivFunc[layerIdx].cfg.relu = activ_func;
    m_state.layerActivFunc[layerIdx].type = State::ActivFuncType::RELU;
    m_state.layerActivFunc[layerIdx].ptr = activation_func_relu;
    m_state.layerActivFunc[layerIdx].derivPtr = activation_func_relu_deriv;
}

void NeuralNet::setActivationFunction(unsigned int layerIdx, ActivationFunction::SIGMOID activ_func)
{
    m_state.layerActivFunc[layerIdx].cfg.sigmoid = activ_func;
    m_state.layerActivFunc[layerIdx].type = State::ActivFuncType::SIGMOID;
    m_state.layerActivFunc[layerIdx].ptr = activation_func_sigmoid;
    m_state.layerActivFunc[layerIdx].derivPtr = activation_func_sigmoid_deriv;
}

void NeuralNet::setActivationFunction(unsigned int layerIdx, ActivationFunction::TANH activ_func)
{
    m_state.layerActivFunc[layerIdx].cfg.tanh = activ_func;
    m_state.layerActivFunc[layerIdx].type = State::ActivFuncType::TANH;
    m_state.layerActivFunc[layerIdx].ptr = activation_func_tanh;
    m_state.layerActivFunc[layerIdx].derivPtr = activation_func_tanh_deriv;
}

void NeuralNet::setActivationFunction(unsigned int layerIdx,
                                      Vector (*activFunc)(const Vector&, unsigned int, StateAccess&),
                                      Vector (*activFuncDeriv)(const Vector&, unsigned int, StateAccess&))
{
    m_state.layerActivFunc[layerIdx].type = State::ActivFuncType::CUSTOM;
    m_state.layerActivFunc[layerIdx].ptr = activFunc;
    m_state.layerActivFunc[layerIdx].derivPtr = activFuncDeriv;
}

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
void NeuralNet::setCostFunction(Vector (*costFunc)(const Vector& output, const Vector& target, StateAccess& net))
{
    m_state.costFunc.type = State::CostFuncType::CUSTOM;
    m_state.costFunc.ptr = costFunc;
}

void NeuralNet::setCostFunction(CostFunction::DIFFERENCE cost_func)
{
    m_state.costFunc.cfg.diff = cost_func;
    m_state.costFunc.type = State::CostFuncType::DIFFERENCE;
    m_state.costFunc.ptr = cost_func_difference;
}

void NeuralNet::setCostFunction(CostFunction::SQUARE_DIFFERENCE cost_func)
{
    m_state.costFunc.cfg.sq_diff = cost_func;
    m_state.costFunc.type = State::CostFuncType::SQUARE_DIFFERENCE;
    m_state.costFunc.ptr = cost_func_square_difference;
}

void NeuralNet::setCostFunction(CostFunction::CROSS_ENTROPY cost_func)
{
    m_state.costFunc.cfg.x_ntrp = cost_func;
    m_state.costFunc.type = State::CostFuncType::CROSS_ENTROPY;
    m_state.costFunc.ptr = cost_func_cross_entropy;
}

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
void NeuralNet::setOptimizeFunction(OptimizeFunction::TEST opt_func)
{
    m_state.optFunc.cfg.test = opt_func;
    m_state.optFunc.type = State::OptFuncType::TEST;
    m_state.optFunc.ptr = optimize_func_backprop;
}

void NeuralNet::setOptimizeFunction(OptimizeFunction::BACKPROP opt_func)
{
    m_state.optFunc.cfg.backprop = opt_func;
    m_state.optFunc.type = State::OptFuncType::BACKPROP;
    m_state.optFunc.ptr = optimize_func_backprop;
}

void NeuralNet::setOptimizeFunction(void (*optFunc)(const Vector&, const Vector&, StateAccess&))
{
    m_state.optFunc.type = State::OptFuncType::CUSTOM;
    m_state.optFunc.ptr = optFunc;
}

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







































































