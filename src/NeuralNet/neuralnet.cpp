#include <NeuralNet/neuralnet.h>
#include <cmath>
#include <random>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <iomanip>

// Initialization function declarations
void init_func_standard(NeuralNet* net);

// Cost function declarations
Vector cost_func_difference(Vector output, Vector target, NeuralNet* net);
Vector cost_func_square_difference(Vector output, Vector target, NeuralNet* net);
Vector cost_func_cross_entropy(Vector output, Vector target, NeuralNet* net);

// Activation function declarations
Vector activation_func_relu(Vector x, NeuralNet* net);
Vector activation_func_sigmoid(Vector x, NeuralNet* net);
Vector activation_func_tanh(Vector x, NeuralNet* net);

// Activation function derivative declarations
Vector activation_func_relu_deriv(Vector x, NeuralNet* net);
Vector activation_func_sigmoid_deriv(Vector x, NeuralNet* net);
Vector activation_func_tanh_deriv(Vector x, NeuralNet* net);

// Optimizer function declarations
void optimize_func_backprop(Vector input, Vector error, NeuralNet* net);

NeuralNet::NeuralNet(std::vector<size_t> sizeVec) :
    m_sizeVec(sizeVec),
    m_batchSize(1),
    m_nEpochs(1),
    m_printInterval(m_batchSize),
    m_softMax(false),
    m_learningRate(1.0)
{
    if (m_sizeVec.size() < 2) {
        THROW_ERROR("Invalid net size: "
                    << m_sizeVec.size()
                    << ", Expected: >=2");
    }

    for (size_t i = 0; i < m_sizeVec.size(); ++i) {
        if (m_sizeVec[i] < 1) {
            THROW_ERROR("Invalid layer size: "
                        << m_sizeVec.size()
                        << ", Expected: >=1");
        }
    }


    // Default initialization function
    setInitializationFunction(InitializationFunction::RANDOM);


    // Default cost function
    setCostFunction(CostFunction::SQUARE_DIFFERENCE);


    // Default optimize function
    setOptimizeFunction(OptimizeFunction::BACKPROP);


    // Create network
    for (size_t i = 0; i < (m_sizeVec.size() - 1); ++i) {

        // Add layer
        m_layers.push_back(Layer(m_sizeVec[i], m_sizeVec[i + 1]));

        // Add activation function to layer, default to sigmoid
        ActivationFunction activFunc;
        activFunc.type = ActivationFunction::SIGMOID;
        activFunc.ptr = activation_func_sigmoid;
        activFunc.derivPtr = activation_func_sigmoid_deriv;
        m_layerActivFunc.push_back(activFunc);
    }


    // Init weights and biases
    m_initFunc.ptr(this);
}

void NeuralNet::propergate(real_vec input)
{
    if (input.size() != m_sizeVec[0]) {
        THROW_ERROR("Invalid input size: "
                    << input.size()
                    << ", Expected: "
                    << m_sizeVec[0]);
    }

    // When not training, dont need to cache weightedSum
    // m_layers[i].output = activFunc(matrixAdd(matrixMulti(input, m_layers[i].weights), m_layers[i].bias));

    // Propergate input layer
    m_layers[0].weightedSum = (m_layers[0].weights * input) + m_layers[0].bias;
    m_layers[0].output = m_layerActivFunc[0].ptr(m_layers[0].weightedSum, this);

    for (unsigned i = 1; i < m_layers.size(); ++i) {
        // Propergate hidden and output layers
        m_layers[i].weightedSum = (m_layers[i].weights * m_layers[i - 1].output) + m_layers[i].bias;
        m_layers[i].output = m_layerActivFunc[i].ptr(m_layers[i].weightedSum, this);
    }

    if (m_softMax) {
        softMax(m_layers.back().output);
    }
}

real NeuralNet::train(real_matrix input, real_matrix target)
{
    // Check sizes
    if (input.size() != target.size()) {
        THROW_ERROR("Invalid size of target vector: "
                    << input.size()
                    << ", Expected: "
                    << m_sizeVec.front());
    }

    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].size() != m_sizeVec.front()) {
            THROW_ERROR("Invalid size of input vector: "
                        << input[i].size()
                        << ", Expected: "
                        << m_sizeVec.front());
        }
    }

    for (size_t i = 0; i < target.size(); ++i) {
        if (target[i].size() != m_sizeVec.back()) {
            THROW_ERROR("Invalid size of target vector: "
                        << target[i].size()
                        << ", Expected: "
                        << m_sizeVec.back());
        }
    }

    // Number of epochs
    for (size_t epoch = 0; epoch < m_nEpochs; ++epoch) {


        // Change print interval at last epoch
        if (epoch == (m_nEpochs - 1)) {
            m_printInterval = 500;
        }


        // Loop training set
        size_t inputIdx = 0;
        while (inputIdx < input.size()) {


            // Training batch
            size_t batchMax = std::min(inputIdx + m_batchSize, input.size());

            Vector error(m_layers.back().size(), 1);

            Vector avg_error(m_layers.back().size(), 0.0);

            Vector avg_input(input.front().size(), 0.0);

            for (size_t batchIdx = inputIdx; batchIdx < batchMax; ++batchIdx) {

                // Propergate input vector
                propergate(input[batchIdx]);

                // Calc error vector, average over batch
                error = m_costFunc.ptr(m_layers.back().output, target[batchIdx], this);

                // Save error to get average
                // Note: avg_error.size() == output.size() == target.size()
                avg_error += error;

                // Average input
                avg_input += input[batchIdx];

//                if ((batchIdx % 10) == 0) {
//                    std::cout << "0" << std::endl;
//                }

//                if ((batchIdx % 100) == 0) {
//                    std::cout << "00" << std::endl;
//                }

//                if ((batchIdx % 1000) == 0) {
//                    std::cout << "00" << std::endl;
//                }

//                if ((batchIdx % 4000) == 0) {
//                    std::cout << "00" << std::endl;
//                }

//                if ((batchIdx % 9000) == 0) {
//                    std::cout << "a" << std::endl;
//                }

//                if ((batchIdx % 9500) == 0) {
//                    std::cout << "b" << std::endl;
//                }

//                if ((batchIdx % 9900) == 0) {
//                    std::cout << "c" << std::endl;
//                }

//                if ((batchIdx % 9999) == 0) {
//                    std::cout << "d" << std::endl;
//                }

                // Print
                if ((batchIdx % m_printInterval) == 0) {
                    printState(input[batchIdx], target[batchIdx], error, batchIdx);
                }
            }


            // Divide to get avgerages
            size_t nSamples = batchMax - inputIdx;
            avg_error /= nSamples;
            avg_input /= nSamples;


            // Optimize on avg_error
            m_optFunc.ptr(avg_input, avg_error, this);


            // Update index
            inputIdx += m_batchSize;
        }
    }

    return 0;
}

//void NeuralNet::backpropergate(real_vec input, real_vec error)
//{
//    // Loop the weights back to front
//    // Loop layers
//    for (int i = (int)m_layers.size() - 1; i >= 0; --i) {

//        // Loop neurons
//        for (size_t j = 0; j < m_layers[i].size(); ++j) {
//            // Current neuron
//            Neuron* currNeuron = &m_layers[i][j];

//            // Calc dC/da aka gradient of neuron a
//            // Different for output layer
//            if ((size_t)i == (m_layers.size() - 1)) {

//                // dC/da = 2 * (a_k - t_k)
//                currNeuron->gradient = error[j]*m_learningRate; // Put learning rate here?
//                // Also, change learningRate according to how much the cost diviates from the mean cost over the last samples
//                // In other words, increase learningRate when encountering outliers

//            } else {

//                currNeuron->gradient = 0;

//                // Loop neurons of the layer to the right of the current layer
//                // dC/da_k1 = sum of (w_k1k2)^L+1 * s'((z_k2)^L+1) * (g_k2)^L+1 for all k2
//                for (size_t k = 0; k < m_layers[i + 1].size(); ++k) {

//                    Neuron* rightNeuron = &m_layers[i + 1][k];

//                    currNeuron->gradient += rightNeuron->weights[j]
//                            * m_layerActivFunc[i].derivPtr(rightNeuron->weightedSum, this)
//                            * rightNeuron->gradient;
//                }
//            }
//        }
//    }

//    // Update weights
//    // Loop layers back to front
//    for (int i = (int)m_layers.size() - 1; i >= 0; --i) {

//        // Loop neurons
//        for (size_t j = 0; j < m_layers[i].size(); ++j) {

//            // Current neuron
//            Neuron* currNeuron = &m_layers[i][j];

//            // Update current neurons bias
//            // dC/db = s'(z) * dC/da
//            // deltaBias = (-1) * dC/db
//            currNeuron->bias -= m_layerActivFunc[i].derivPtr(currNeuron->weightedSum, this)
//                    * currNeuron->gradient;

//            // Loop weights
//            for (size_t k = 0; k < currNeuron->size(); ++k) {

//                // Weights in the first layer are connected to the input vector and NOT a neuron layer
//                if (i == 0) {

//                    // dC/d(w_k1k2)^L = input[k1] * s'((z_k2)^L) * dC/d(a_k2)^L
//                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
//                    currNeuron->weights[k] -= input[k]
//                            * m_layerActivFunc[i].derivPtr(currNeuron->weightedSum, this)
//                            * currNeuron->gradient;
//                } else {

//                    // Neuron in the layer to the left that the weight is connected to
//                    Neuron* leftNeuron = &m_layers[i - 1][k];

//                    // dC/d(w_k1k2)^L = (a_k1)^L-1 * s'((z_k2)^L) * dC/d(a_k2)^L
//                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
//                    currNeuron->weights[k] -= leftNeuron->output
//                            * m_layerActivFunc[i].derivPtr(currNeuron->weightedSum, this)
//                            * currNeuron->gradient;
//                }
//            }
//        }
//    }
//}

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

void NeuralNet::printState(Vector input, Vector target, Vector error, size_t batchIdx)
{
    // Create stream
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);



    // Index
    ss << "idx: [ " << batchIdx << " ]\n";



    // Learning rate
    ss << "lrt: [ " << std::setprecision(4) << m_learningRate << std::setprecision(2) << " ]\n";



    // Input
//    ss << "inp: [ " << input[0];

//    for (size_t i = 1; i < input.size(); ++i) {
//        ss << ",  " << input[i];
//    }

//    ss<< " ]\n";



    // Output
    real output = m_layers.back().output(0);

    if (output < 0) {
        ss << "out: [" << output;
    } else {
        ss << "out: [ " << output;
    }

    for (size_t i = 1; i < m_layers.back().size(); ++i) {
        output = m_layers.back().output(i);

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

// Getters and Setters
std::vector<size_t> NeuralNet::sizeVec()
{
    return m_sizeVec;
}

std::vector<Layer>& NeuralNet::layers()
{
    return m_layers;
}

unsigned int NeuralNet::batchSize()
{
    return m_batchSize;
}

void NeuralNet::setBatchSize(unsigned int batchSize)
{
    m_batchSize = batchSize;
}

unsigned int NeuralNet::nEpochs()
{
    return m_nEpochs;
}

void NeuralNet::setNEpochs(unsigned int nEpochs)
{
    m_nEpochs = nEpochs;
}

unsigned int NeuralNet::printInterval()
{
    return m_printInterval;
}

void NeuralNet::setPrintInterval(unsigned int printInterval)
{
    m_printInterval = printInterval;
}

void NeuralNet::setSoftMax(bool softMax)
{
    m_softMax = softMax;
}

real NeuralNet::learningRate()
{
    return m_learningRate;
}

void NeuralNet::setLearningRate(real learningRate)
{
    m_learningRate = learningRate;
}

// Initialization functions
void NeuralNet::setInitializationFunction(InitializationFunction::Type init_func)
{
    switch (init_func) {
        case InitializationFunction::ALL_ZERO:
            m_initFunc.ptr = init_func_standard;
            break;

        case InitializationFunction::RANDOM:
            m_initFunc.ptr = init_func_standard;
            break;
    }

    m_initFunc.type = init_func;
}

void NeuralNet::setInitializationFunction(void (*initFunc)(NeuralNet *))
{
    m_initFunc.type = 0;
    m_initFunc.ptr = initFunc;
}

void init_func_standard(NeuralNet* net)
{
    // Random generator
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<real> d{0, 1};

    // Get layers
    std::vector<Layer>& layers = net->layers();

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

// Cost functions
void NeuralNet::setCostFunction(CostFunction::Type cost_func)
{
    switch (cost_func) {
        case CostFunction::DIFFERENCE:
            m_costFunc.ptr = cost_func_difference;
            break;

        case CostFunction::SQUARE_DIFFERENCE:
            m_costFunc.ptr= cost_func_square_difference;
            break;

        case CostFunction::CROSS_ENTROPY:
            m_costFunc.ptr = cost_func_cross_entropy;
            break;
    }

    m_costFunc.type = cost_func;
}

void NeuralNet::setCostFunction(Vector (*costFunc)(Vector output, Vector target, NeuralNet* net))
{
    m_costFunc.type = 0;
    m_costFunc.ptr = costFunc;
}

Vector cost_func_difference(Vector output, Vector target, NeuralNet *net)
{
    (void)net;

    return output - target;
}

Vector cost_func_square_difference(Vector output, Vector target, NeuralNet *net)
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

Vector cost_func_cross_entropy(Vector output, Vector target, NeuralNet *net)
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

// Activation functions
Vector NeuralNet::activationFunction(unsigned int layerIdx, Vector x)
{
    return m_layerActivFunc[layerIdx].ptr(x, this);
}

Vector NeuralNet::activationFunctionDerivate(unsigned int layerIdx, Vector x)
{
    return m_layerActivFunc[layerIdx].derivPtr(x, this);
}

void NeuralNet::setHiddenLayerActivationFunction(ActivationFunction::Type activ_func)
{
    for (size_t i = 0; i < m_layers.size() - 1; ++i) {
        setActivationFunction(i, activ_func);
    }
}

void NeuralNet::setHiddenLayerActivationFunction(Vector (*activFunc)(Vector, NeuralNet*), Vector (*activFuncDeriv)(Vector, NeuralNet*))
{
    for (size_t i = 0; i < m_layers.size() - 1; ++i) {
        setActivationFunction(i, activFunc, activFuncDeriv);
    }
}

void NeuralNet::setOutputLayerActivationFunction(ActivationFunction::Type activ_func)
{
    setActivationFunction(m_layers.size() - 1, activ_func);
}

void NeuralNet::setOutputLayerActivationFunction(Vector (*activFunc)(Vector, NeuralNet*), Vector (*activFuncDeriv)(Vector, NeuralNet*))
{
    // What happens at layer.size == 1???
    setActivationFunction(m_layers.size() - 1, activFunc, activFuncDeriv);
}

void NeuralNet::setActivationFunction(unsigned int layerIdx, ActivationFunction::Type activ_func)
{
    switch (activ_func) {
        case ActivationFunction::RELU:
            m_layerActivFunc[layerIdx].ptr = activation_func_relu;
            m_layerActivFunc[layerIdx].derivPtr = activation_func_relu_deriv;
            break;

        case ActivationFunction::SIGMOID:
            m_layerActivFunc[layerIdx].ptr = activation_func_sigmoid;
            m_layerActivFunc[layerIdx].derivPtr = activation_func_sigmoid_deriv;
            break;

        case ActivationFunction::TANH:
            m_layerActivFunc[layerIdx].ptr = activation_func_tanh;
            m_layerActivFunc[layerIdx].derivPtr = activation_func_tanh_deriv;
            break;
    }

    m_layerActivFunc[layerIdx].type = activ_func;
}

void NeuralNet::setActivationFunction(unsigned int layerIdx, Vector (*activFunc)(Vector, NeuralNet*), Vector (*activFuncDeriv)(Vector, NeuralNet*))
{
    m_layerActivFunc[layerIdx].type = 0;
    m_layerActivFunc[layerIdx].ptr = activFunc;
    m_layerActivFunc[layerIdx].derivPtr = activFuncDeriv;
}

Vector activation_func_relu(Vector x, NeuralNet *net)
{
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

Vector activation_func_sigmoid(Vector x, NeuralNet *net)
{
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

Vector activation_func_tanh(Vector x, NeuralNet *net)
{
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
Vector activation_func_relu_deriv(Vector x, NeuralNet *net)
{
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

Vector activation_func_sigmoid_deriv(Vector x, NeuralNet *net)
{
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

Vector activation_func_tanh_deriv(Vector x, NeuralNet *net)
{
    (void)net;

    Vector res(x.size(), 1);

    for (unsigned i = 0; i < x.size(); ++i) {
        res(i) = 2 / (std::cosh(2*x(i)) + 1);
    }

    return res;
}

// Optimize functions
void NeuralNet::setOptimizeFunction(OptimizeFunction::Type opt_func)
{
    switch (opt_func) {
        case OptimizeFunction::TEST:
            m_optFunc.ptr = optimize_func_backprop;
            break;

        case OptimizeFunction::BACKPROP:
            m_optFunc.ptr= optimize_func_backprop;
            break;
    }

    m_optFunc.type = opt_func;
}

void NeuralNet::setOptimizeFunction(void (*optFunc)(Vector, Vector, NeuralNet*))
{
    m_optFunc.type = 0;
    m_optFunc.ptr = optFunc;
}

void optimize_func_backprop(Vector input, Vector error, NeuralNet* net)
{
    std::vector<Layer>& layers = net->layers();

    // Backprop output layer
    layers.back().gradient = error*net->learningRate();

    // Loop backwards
    for (int i = (layers.size() - 2); i >= 0; --i) {

//                layers[i].gradient = layers[i + 1].weights.transpose()
//                        * (net->activationFunctionDerivate(i, layers[i + 1].weightedSum).multiplyElemWise(layers[i + 1].gradient));

        layers[i].gradient = layers[i + 1].weights.transpose() * (net->activationFunctionDerivate(i, layers[i + 1].weightedSum).multiplyElemWise(layers[i + 1].gradient));
    }

    // Loop backwards
    for (int i = (layers.size() - 1); i > 0; --i) {

//        layers[i].weights = layers[i].weights.subtractElemWise(
//                    (net->activationFunctionDerivate(i, layers[i].weightedSum) * layers[i].gradient).matMul(layers[i - 1].output)
//                );

        layers[i].weights -= net->activationFunctionDerivate(i, layers[i].weightedSum).multiplyElemWise(layers[i].gradient) * layers[i - 1].output.transpose();


//        Matrix mat1 = net->activationFunctionDerivate(i, layers[i].weightedSum);

//        Matrix mat2 = layers[i].gradient;

//        Matrix mat3 = mat1.multiplyElemWise(mat2);

//        Matrix mat4 = layers[i - 1].output.transpose();

//        Matrix mat5 = mat3 * mat4;

//        layers[i].weights = layers[i].weights - mat5;
    }

//    layers[0].weights = layers[0].weights.subtractElemWise(
//                (net->activationFunctionDerivate(0, layers[0].weightedSum) * layers[0].gradient).matMul(input)
//            );

    layers[0].weights -= net->activationFunctionDerivate(0, layers[0].weightedSum).multiplyElemWise(layers[0].gradient) * input.transpose();

}







































































