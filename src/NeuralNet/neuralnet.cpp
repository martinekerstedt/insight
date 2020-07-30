#include <NeuralNet/neuralnet.h>
#include <NeuralNet/neuron.h>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <iomanip>

NeuralNet::NeuralNet(std::vector<size_t> size_vec) :
    batchSize(1),
    nEpochs(1),
    printInterval(batchSize),
    softMaxOutput(false),
    costFunction(CostFunction::DIFFERENCE),
    learningRate(1.0),
    momentum(0.0),
    m_size_vec(size_vec),
    m_eulerConstant(static_cast<real>(std::exp(1.0)))
{
    if (size_vec.size() < 2)
    {
        THROW_ERROR("Invalid net size: "
                    << size_vec.size()
                    << ", Expected: >=2");
    }

    for (size_t i = 0; i < size_vec.size(); ++i)
    {
        if (size_vec[i] < 1)
        {
            THROW_ERROR("Invalid layer size: "
                        << size_vec.size()
                        << ", Expected: >=1");
        }
    }

    // Total number of layers in the net
    size_t nLayers = size_vec.size() - 1;

    // Random generator
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<real> d{0, 1};

    // All layers have neurons with size
    // equal number of neurons in previous layer   
    for (size_t i = 0; i < nLayers; ++i)
    {
        if (nLayers == 1) {
               layers.push_back(Layer(size_vec[i + 1], Layer::Type::INPUT_OUTPUT, Layer::Activation::SIGMOID));
        } else {
            if (i == 0) {
                layers.push_back(Layer(size_vec[i + 1], Layer::Type::INPUT, Layer::Activation::SIGMOID));
            } else if (i == (nLayers - 1)) {
                layers.push_back(Layer(size_vec[i + 1], Layer::Type::OUTPUT, Layer::Activation::SIGMOID));
            } else {
                layers.push_back(Layer(size_vec[i + 1], Layer::Type::HIDDEN, Layer::Activation::SIGMOID));
            }
        }


        for (size_t j = 0; j < layers.back().size; ++j)
        {



            layers.back().neurons.push_back(Neuron(size_vec[i]));



            Neuron* currNeuron = &layers.back().neurons.back();

            for (size_t k = 0; k < currNeuron->size; ++k)
            {
//                currNeuron->weights.push_back(d(gen)
//                                              * std::sqrt(2 / (real)currNeuron->size));

                 currNeuron->weights.push_back(d(gen));

//                currNeuron->weights.push_back(-1.0
//                                              + (2 * (((real)rand()) / ((real)RAND_MAX))));
            }
        }
    }

    // Set size of error vector
    m_error.resize(layers.back().size, 0);
}

void NeuralNet::propergate(real_vec input)
{
    if (input.size() != m_size_vec[0])
    {
        THROW_ERROR("Invalid input size: "
                    << input.size()
                    << ", Expected: "
                    << m_size_vec[0]);
    }

    real_vec res;

    for (size_t i = 0; i < layers.size(); ++i)
    {
        for (size_t j = 0; j < layers[i].size; ++j)
        {
            // Calc neuron output
            Neuron* neuron = &layers[i][j];

            neuron->weightedSum = 0;

            // Weighted sum
            for (size_t k = 0; k < neuron->size; ++k)
            {
                neuron->weightedSum += input[k] * neuron->weights[k];
            }

            // Add bias
            neuron->weightedSum += neuron->bias;

            // Apply activation func
            neuron->output = activationFunction(layers[i].activation, neuron->weightedSum);

            // Save output to feed to next layer
            res.push_back(neuron->output);
        }

        input = res;
        res.clear();                
    }

    if (softMaxOutput) {
        softMax(layers.back());
    }
}

void NeuralNet::calcError(real_vec target)
{
    if (target.size() != m_size_vec.back()) {
        THROW_ERROR("Invalid size of target vector: "
                    << target.size()
                    << ", Expected: "
                    << m_size_vec.back());
    }

    for (size_t i = 0; i < target.size(); ++i) {
        m_error[i] = costFunc(layers.back()[i].output, target[i]);
    }
}

real NeuralNet::train(real_matrix input, real_matrix target)
{
    // Check sizes
    if (input.size() != target.size()) {
        THROW_ERROR("Invalid size of target vector: "
                    << input.size()
                    << ", Expected: "
                    << m_size_vec.front());
    }    

    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].size() != m_size_vec.front()) {
            THROW_ERROR("Invalid size of input vector: "
                        << input[i].size()
                        << ", Expected: "
                        << m_size_vec.front());
        }
    }

    for (size_t i = 0; i < target.size(); ++i) {
        if (target[i].size() != m_size_vec.back()) {
            THROW_ERROR("Invalid size of target vector: "
                        << target[i].size()
                        << ", Expected: "
                        << m_size_vec.back());
        }
    }


    // Number of epochs
    for (size_t epoch = 0; epoch < nEpochs; ++epoch) {


        // Change print interval at last epoch
        if (epoch == (nEpochs - 1)) {
            printInterval = 500;
        }


        // Loop training set
        size_t inputIdx = 0;
        while (inputIdx < input.size()) {



            // Training batch
            size_t batchMax = std::min(inputIdx + batchSize, input.size());

            real_vec avg_error;
            avg_error.resize(target.front().size(), 0);

            real_vec avg_input;
            avg_input.resize(input.front().size(), 0);

            for (size_t batchIdx = inputIdx; batchIdx < batchMax; ++batchIdx) {

                // Propergate input vector
                propergate(input[batchIdx]);

                // Calc error vector
                calcError(target[batchIdx]);

                // Calc error given output and target
                // Note: avg_error.size() == output.size() == target.size()
                for (size_t errorIdx = 0; errorIdx < avg_error.size(); ++errorIdx) {
                    avg_error[errorIdx] += m_error[errorIdx];
                }

                // Avg input
                for (size_t i = 0; i < avg_input.size(); ++i) {
                    avg_input[i] += input[batchIdx][i];
                }

                // Print
                if ((batchIdx % printInterval) == 0) {
                    printState(input[batchIdx], target[batchIdx], batchIdx);
                }
            }



            // Divide to get avg
            size_t nSamples = batchMax - inputIdx;

            for (size_t errorIdx = 0; errorIdx < avg_error.size(); ++errorIdx) {
                avg_error[errorIdx] /= nSamples;
            }

            for (size_t i = 0; i < avg_input.size(); ++i) {
                avg_input[i] /= nSamples;
            }



            // Update learning rate
//            real cost = 0;

//            for (size_t i = 0; i < avg_error.size(); ++i) {
//                cost += std::pow(avg_error[i], 2);
//            }

//            updateLearningRate(cost);



            // Backprop on avg_error
            backpropergate(avg_input, avg_error);



            // Update index
            inputIdx += batchSize;
        }
    }

    return 0;
}

void NeuralNet::backpropergate(real_vec input, real_vec error)
{
    // Loop the weights back to front
    // Loop layers
    for (int i = (int)layers.size() - 1; i >= 0; --i)
    {
        // Loop neurons
        for (size_t j = 0; j < layers[i].size; ++j)
        {
            // Current neuron
            Neuron* currNeuron = &layers[i][j];

            // Calc dC/da aka gradient of neuron a
            // Different for output layer
            if ((layers[i].type == Layer::Type::OUTPUT) || (layers[i].type == Layer::Type::INPUT_OUTPUT))
            {
                // dC/da = 2 * (a_k - t_k)
                currNeuron->gradient = 2*error[j]*learningRate; // Put learning rate here?
                // Also, change learningRate according to how much the cost diviates from the mean cost over the last samples
                // In other words, increase learningRate when encountering outliers
            }
            else
            {
                currNeuron->gradient = 0;

                // Loop neurons of the layer to the right of the current layer
                // dC/da_k1 = sum of (w_k1k2)^L+1 * s'((z_k2)^L+1) * (g_k2)^L+1 for all k2
                for (size_t k = 0; k < layers[i + 1].size; ++k)
                {
                    Neuron* rightNeuron = &layers[i + 1][k];

                    currNeuron->gradient += rightNeuron->weights[j]
                            * activationFunctionDerivative(layers[i].activation,
                                                           rightNeuron->weightedSum)
                            * rightNeuron->gradient;
                }
            }
        }
    }

    // Update weights
    // Loop layers back to front
    for (int i = (int)layers.size() - 1; i >= 0; --i)
    {
        // Loop neurons
        for (size_t j = 0; j < layers[i].size; ++j)
        {
            // Current neuron
            Neuron* currNeuron = &layers[i][j];

            // Update current neurons bias
            // dC/db = s'(z) * dC/da
            // deltaBias = (-1) * dC/db
            currNeuron->bias -= activationFunctionDerivative(layers[i].activation,
                                                             currNeuron->weightedSum)
                    * currNeuron->gradient;

            // Loop weights
            for (size_t k = 0; k < currNeuron->size; ++k)
            {
                // Weights in the first layer are connected to the input vector and NOT a neuron layer
                if ((layers[i].type == Layer::Type::INPUT) || (layers[i].type == Layer::Type::INPUT_OUTPUT))
                {
                    // dC/d(w_k1k2)^L = input[k1] * s'((z_k2)^L) * dC/d(a_k2)^L
                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
                    currNeuron->weights[k] -= input[k]
                            * activationFunctionDerivative(layers[i].activation,
                                                           currNeuron->weightedSum)
                            * currNeuron->gradient;
                }
                else
                {

                    // Neuron in the layer to the left that the weight is connected to
                    Neuron* leftNeuron = &layers[i - 1][k];

                    // dC/d(w_k1k2)^L = (a_k1)^L-1 * s'((z_k2)^L) * dC/d(a_k2)^L
                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
                    currNeuron->weights[k] -= leftNeuron->output
                            * activationFunctionDerivative(layers[i].activation,
                                                           currNeuron->weightedSum)
                            * currNeuron->gradient;
                }
            }
        }
    }
}

std::vector<size_t> NeuralNet::size_vec()
{
    return m_size_vec;
}

real NeuralNet::costFunc(real output, real target)
{
    if (costFunction == CostFunction::DIFFERENCE) {

        return output - target;

    } else if (costFunction == CostFunction::CROSS_ENTROPY) {

        // Assumes that target is either 1 or 0
        // And that output is between 0 and 1
        if ((output > 1.0) || (output < 0.0)) {
            THROW_ERROR("Invalid value of output: "
                        << output
                        << ", Expected: Between 0 and 1");
        }

        if ((target > 1.0) || (target < 0.0)) {
            THROW_ERROR("Invalid value of target: "
                        << output
                        << ", Expected: Between 0 and 1");
        }

        if (target == 1.0) {
            return -std::log(output);
        } else {
            return -std::log(1 - output);
        }

    } else if (costFunction == CostFunction::SQUARE_DIFFERENCE) {

        real diff = output - target;

        if (diff >= 0.0) {
            return std::pow(diff, 2);
        } else {
            return -std::pow(diff, 2);
        }

    } else {
        // Error
        THROW_ERROR("Invalid value of costFunction: "
                    << (int)costFunction
                    << ", Expected: value of type NeuralNet::CostFunction");
    }
}

real NeuralNet::activationFunction(Layer::Activation func, real x)
{
    if (func == Layer::Activation::RELU) {

        // Relu
        if (x >= 0.0) {
            return x;
        } else {
            if (x < -15.0) {
                return -0.2;
            } else {
                return 0.2*(std::pow(m_eulerConstant, x) - 1.0);
            }
        }

    } else if (func == Layer::Activation::SIGMOID) {

        // Sigmoid
        if (x > 15.0) {
            return 1.0;
        } else if (x < -15.0) {
            return 0.0;
        } else {
            return 1.0 / (1.0 + std::pow(m_eulerConstant, -x));
        }

    } else if (func == Layer::Activation::TANH) {

        // Tanh
        if (x > 15.0) {
            return 1.0;
        } else if (x < -15.0) {
            return -1.0;
        } else {
            return std::tanh(x);
        }

    } else {
        THROW_ERROR("Invalid value of func: "
                    << (int)func
                    << ", Expected: value of type NeuralNet::Layer::Activation");
    }
}

real NeuralNet::activationFunctionDerivative(Layer::Activation func, real x)
{
    if (func == Layer::Activation::RELU) {

        // Relu derivative
        if (x >= 0) {
            return 1.0;
        } else {
            if (x < -15.0) {
                return 0;
            } else {
                return 0.2*std::pow(m_eulerConstant, x);
            }
        }

    } else if (func == Layer::Activation::SIGMOID) {

        // Sigmoid derivative
        if (x > 15) {
            return 0;
        } else if (x < -15) {
            return 0;
        } else {
            return std::pow(m_eulerConstant, -x) / std::pow(1 + std::pow(m_eulerConstant, -x), 2);
        }

    } else if (func == Layer::Activation::TANH) {

        // Tanh derivative
        return 2 / (std::cosh(2*x) + 1);

    } else {
        THROW_ERROR("Invalid value of func: "
                    << (int)func
                    << ", Expected: value of type NeuralNet::Layer::Activation");
    }
}

void NeuralNet::softMax(real_vec& vec)
{
    real sum = 0;

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = std::exp(vec[i]);
        sum += vec[i];
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] /= sum;
    }
}

void NeuralNet::softMax(Layer& layer)
{
    real sum = 0;

    for (size_t i = 0; i < layer.size; ++i) {
        layer[i].output = std::exp(layer[i].output);
        sum += layer[i].output;
    }

    for (size_t i = 0; i < layer.size; ++i) {
        layer[i].output /= sum;
    }
}

void NeuralNet::updateLearningRate(real cost)
{
    if (cost > 0.8) {
        learningRate = 0.008;
    } else if (cost <= 0.0) {
        learningRate = 0.0;
    } else {
        // 0.05*(x^2 + x)
        learningRate = 0.008*(cost / 0.8);
    }
}

void NeuralNet::setHiddenActivation(Layer::Activation activation)
{
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        layers[i].activation = activation;
    }
}

void NeuralNet::setOutputActivation(Layer::Activation activation)
{
    layers.back().activation = activation;
}

void NeuralNet::printState(real_vec input, real_vec target, size_t batchIdx)
{
    // Create stream
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);



    // Index
    ss << "idx: [ " << batchIdx << " ]\n";



    // Learning rate
    ss << "lrt: [ " << std::setprecision(4) << learningRate << std::setprecision(2) << " ]\n";



    // Input
//    ss << "inp: [ " << input[0];

//    for (size_t i = 1; i < input.size(); ++i) {
//        ss << ",  " << input[i];
//    }

//    ss<< " ]\n";



    // Output
    real output = layers.back()[0].output;

    if (output < 0) {
        ss << "out: [" << output;
    } else {
        ss << "out: [ " << output;
    }

    for (size_t i = 1; i < layers.back().size; ++i) {
        output = layers.back()[i].output;

        if (output < 0) {
            ss << ", " << output;
        } else {
            ss << ",  " << output;
        }
    }

    ss << " ]\n";



    // Target
    ss << "tar: [ " << target[0];

    for (size_t i = 1; i < target.size(); ++i) {
        ss << ",  " << target[i];
    }

    ss << " ]\n";



    // Error
    real error = m_error[0];

    if (error < 0) {
        ss << "err: [" << error;
    } else {
        ss << "err: [ " << error;
    }

    for (size_t i = 1; i < m_error.size(); ++i) {
        error = m_error[i];

        if (error < 0) {
            ss << ", " << error;
        } else {
            ss << ",  " << error;
        }
    }

    ss << " ]\n";



    // Cost
    real cost = 0;
    for (size_t i = 0; i < m_error.size(); ++i) {
        cost += std::pow(m_error[i], 2);
    }

    if (cost < 0) {
        ss << "cst: [" << cost << " ]\n";
    } else {
        ss << "cst: [ " << cost << " ]\n";
    }



    // Print stream
    std::cout << ss.str() << std::endl;
}





































































































































//real NeuralNet::train(real_vec input, real_vec target, real_vec* error)
//{
//    if (target.size() != m_size_vec.back())
//    {
//        THROW_ERROR("Invalid size of target vector: "
//                    + std::to_string(target.size())
//                    + ", Expected: "
//                    + std::to_string(m_size_vec.back()));
//    }

//    if (input.size() != m_size_vec.front())
//    {
//        THROW_ERROR("Invalid size of target vector: "
//                    + std::to_string(input.size())
//                    + ", Expected: "
//                    + std::to_string(m_size_vec.front()));
//    }

//    // Propergate
//    real_vec res = propergate(input);

//    // Error
//    // real_vec error;

//    real cost = 0;

//    error->clear();

//    for (size_t i = 0; i < res.size(); ++i)
//    {
//        error->push_back(std::pow(res[i] - target[i], 2));
//        cost += std::pow(res[i] - target[i], 2);
//    }

//    // Loop the weights back to front
//    // Loop layers
//    for (int i = (int)layers.size() - 1; i >= 0; --i)
//    {
//        // Loop neurons
//        for (size_t j = 0; j < layers[i].size; ++j)
//        {
//            // Current neuron
//            Neuron* currNeuron = &layers[i][j];

//            // Calc dC/da aka gradient of neuron a
//            // Different for output layer
//            if (layers[i].type == Layer::Type::OUTPUT)
//            {
//                // dC/da = 2 * (a_k - t_k)
//                currNeuron->gradient = 2*(currNeuron->output - target[j]);
//            }
//            else
//            {
//                currNeuron->gradient = 0;

//                // Loop neurons of the layer to the right of the current layer
//                // dC/da_k1 = sum of (w_k1k2)^L+1 * s'((z_k2)^L+1) * (g_k2)^L+1 for all k2
//                for (size_t k = 0; k < layers[i + 1].size; ++k)
//                {
//                    Neuron* rightNeuron = &layers[i + 1][k];

//                    currNeuron->gradient += rightNeuron->weights[j]
//                            * activationFunctionDerivative(layers[i].activation,
//                                                           rightNeuron->weightedSum)
//                            * rightNeuron->gradient;
//                }
//            }
//        }
//    }

//    // Update weights
//    // Loop layers back to front
//    for (int i = (int)layers.size() - 1; i >= 0; --i)
//    {
//        // Loop neurons
//        for (size_t j = 0; j < layers[i].size; ++j)
//        {
//            // Current neuron
//            Neuron* currNeuron = &layers[i][j];

//            // Update current neurons bias
//            // dC/db = s'(z) * dC/da
//            // deltaBias = (-1) * dC/db
//            currNeuron->bias -= m_learningRate
//                                * activationFunctionDerivative(layers[i].activation,
//                                                               currNeuron->weightedSum)
//                                * currNeuron->gradient;

//            // Loop weights
//            for (size_t k = 0; k < currNeuron->size; ++k)
//            {
//                // Weights in the first layer are connected to the input vector and NOT a neuron layer
//                if (layers[i].type == Layer::Type::INPUT)
//                {
//                    // dC/d(w_k1k2)^L = input[k1] * s'((z_k2)^L) * dC/d(a_k2)^L
//                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
//                    currNeuron->weights[k] -= m_learningRate
//                                                * input[k]
//                                                * activationFunctionDerivative(layers[i].activation,
//                                                                               currNeuron->weightedSum)
//                                                * currNeuron->gradient;
//                }
//                else
//                {

//                    // Neuron in the layer to the left that the weight is connected to
//                    Neuron* leftNeuron = &layers[i - 1][k];

//                    // dC/d(w_k1k2)^L = (a_k1)^L-1 * s'((z_k2)^L) * dC/d(a_k2)^L
//                    // deltaWeight = (-1) * dC/d(w_k1k2)^L
//                    currNeuron->weights[k] -= m_learningRate
//                                                * leftNeuron->output
//                                                * activationFunctionDerivative(layers[i].activation,
//                                                                               currNeuron->weightedSum)
//                                                * currNeuron->gradient;
//                }
//            }
//        }
//    }

//    // Return error vec
//    // return error;

//    return cost;
//}

//real NeuralNet::neuronOutput(Neuron* neuron, real_vec input, Layer::Activation activation)
//{
//    if (neuron->size != input.size())
//    {
//        THROW_ERROR("Invalid input size: " + std::to_string(input.size()));
//    }

//    neuron->weightedSum = 0;

//    // Weighted sum
//    for (size_t i = 0; i < neuron->size; ++i)
//    {
//        neuron->weightedSum += input[i] * neuron->weights[i];
//    }

//    // Add bias
//    neuron->weightedSum += neuron->bias;

//    // Apply activation func
//    neuron->output = activationFunction(activation, neuron->weightedSum);

//    return neuron->output;
//}

//real_vec NeuralNet::propergate(real_vec input)
//{
//    if (input.size() != m_size_vec[0])
//    {
//        THROW_ERROR("Invalid input size: "
//                    + std::to_string(input.size())
//                    + ", Expected: "
//                    + std::to_string(m_size_vec[0]));
//    }

//    real_vec res;

//    for (size_t i = 0; i < layers.size(); ++i)
//    {
//        for (size_t j = 0; j < layers[i].size; ++j)
//        {
//            res.push_back(neuronOutput(&layers[i][j], input, layers[i].activation));
//        }

//        input = res;
//        res.clear();
//    }

//    return input;
//}

//real NeuralNet::relu(real x)
//{
//    if (x >= 0) {
//        return x;
//    } else {
//        return 0;
//    }
//}

//real NeuralNet::reluDerivative(real x)
//{
//    if (x >= 0) {
//        return 1.0;
//    } else {
//        return 0;
//    }
//}

//real NeuralNet::sigmoid(real x)
//{
//    if (x > 10) {
//        return 1;
//    } else if (x < -10) {
//        return 0;
//    } else {
//        return 1 / (1 + std::pow(m_eulerConstant, -x));
//    }
//}

//real NeuralNet::sigmoidDerivative(real x)
//{
//    if (x > 10) {
//        return 0;
//    } else if (x < -10) {
//        return 0;
//    } else {
//        return std::pow(m_eulerConstant, -x) / std::pow(1 + std::pow(m_eulerConstant, -x), 2);
//    }
//}

//real NeuralNet::tanh(real x)
//{
//    if (x > 10) {
//        return 1;
//    } else if (x < -10) {
//        return -1;
//    } else {
//        return std::tanh(x);
//    }
//}

//real NeuralNet::tanhDerivative(real x)
//{
//    return 2 / (std::cosh(2*x) + 1);
//}

//real (NeuralNet::*(NeuralNet::getFuncPtr(Neuron::Activation func)))(real x)
//{
//    if (func == Neuron::Activation::RELU) {
//        return &NeuralNet::relu;
//    } else if (func == Neuron::Activation::SIGMOID) {
//        return &NeuralNet::sigmoid;
//    } else if (func == Neuron::Activation::TANH) {
//        return &NeuralNet::tanh;
//    }

//    // Error
//    return &NeuralNet::relu;
//}

//void NeuralNet::activationFunction(ActivationFunc func)
//{
//    if (func == ActivationFunc::RELU) {
//        m_afs = &NeuralNet::relu;
//        m_afsd = &NeuralNet::reluDerivative;
//    } else if (func == ActivationFunc::SIGMOID) {
//        m_afs = &NeuralNet::sigmoid;
//        m_afsd = &NeuralNet::sigmoidDerivative;
//    } else if (func == ActivationFunc::TANH) {
//        m_afs = &NeuralNet::tanh;
//        m_afsd = &NeuralNet::tanhDerivative;
//    } else {
//        // Error
//    }

//    m_activationFunc = func;
//}

// real NeuralNet::m_afs(real x)
// {
//     return (x / (2*(1 + std::abs(x)))) + 0.5;
// }

// real NeuralNet::m_afsd(real x)
// {
//     return 1 / (2*std::pow(1 + std::abs(x), 2));
// }

// real NeuralNet::m_afs(real x)
// {
//     const real a = 0.1;

//     if (x >= 0)
//     {
//         if (x > 1.0)
//         {
//             return 1.0;
//         }
//         else
//         {
//             return x;
//         }
//     }
//     else
//     {
//         if (x < -100.0)
//         {
//             return -a;
//         }
//         else
//         {
//             return a*(std::pow(m_eulerConstant, x) - 1.0);
//         }
//     }
// }

// real NeuralNet::m_afsd(real x)
// {
//     const real a = 0.1;

//     if (x >= 0)
//     {
//         if (x > 1.0)
//         {
//             return 0.0;
//         }
//         else
//         {
//             return 1.0;
//         }
//     }
//     else
//     {
//         if (x < -100.0)
//         {
//             return 0.0;
//         }
//         else
//         {
//             return a*(std::pow(m_eulerConstant, x));
//         }
//     }
// }

// real NeuralNet::m_afs(real x)
// {
//     if (x >= 0)
//     {
//         if (x > 50.0)
//         {
//             return 1.0;
//         }
//         else
//         {
//             return 1 / (1 + std::pow(m_eulerConstant, -x));
//         }
//     }
//     else
//     {
//         if (x < -50.0)
//         {
//             return 0.0;
//         }
//         else
//         {
//             return 1 / (1 + std::pow(m_eulerConstant, -x));
//         }
//     }
// }

// real NeuralNet::m_afsd(real x)
// {
//     if (x >= 0)
//     {
//         if (x > 50.0)
//         {
//             return 0.0;
//         }
//         else
//         {
//             return std::pow(m_eulerConstant, -x) / std::pow(1 + std::pow(m_eulerConstant, -x), 2);
//         }
//     }
//     else
//     {
//         if (x < -50.0)
//         {
//             return 0.0;
//         }
//         else
//         {
//             return std::pow(m_eulerConstant, -x) / std::pow(1 + std::pow(m_eulerConstant, -x), 2);
//         }
//     }
// }
