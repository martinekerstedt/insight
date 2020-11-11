#include <NeuralNet/neuralnet.h>
#include <NeuralNet/functions.h>
#include <cmath>
//#include <random>
//#include <algorithm>

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace NeuralNet;

Model::Model() :
    Model({1, 2, 1})
{

}

Model::Model(const std::initializer_list<size_t>& list) :
    Model(std::vector<size_t>(list))
{

}

Model::Model(const std::vector<size_t>& sizeVec) :
    m_context(m_config, m_state)
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

    // Config
    m_config.sizeVec = sizeVec;
    m_config.batchSize = 1;
    m_config.printInterval = 1;


    // Step
    m_state.step = 0;


    // Default initialization function
    InitializationFunction::RANDOM_NORMAL rnd_cfg;
    setInitializationFunction(rnd_cfg);


    // Default cost function
    CostFunction::SQUARE_DIFFERENCE sq_diff_cfg;
    setCostFunction(sq_diff_cfg);


    // Default optimize function
    OptimizeFunction::BACKPROP backprop_cfg;
    setOptimizeFunction(backprop_cfg);


    // Initialize error vectors
    m_state.error.resize(sizeVec.back(), 0.0);
    m_state.avg_error.resize(sizeVec.back(), 0.0);


    // Initialize average input
    m_state.avg_input.resize(sizeVec.front(), 0.0);


    // Create network
    for (size_t i = 0; i < (sizeVec.size() - 1); ++i) {

        // Add layer
        m_state.layers.push_back(State::Layer(sizeVec[i], sizeVec[i + 1]));
        Config::ActivFunc activFunc;
        activFunc.type = Config::ActivFuncType::SIGMOID;
        activFunc.ptr = activation_func_sigmoid;
        activFunc.derivPtr = activation_func_sigmoid_deriv;
        ActivationFunction::SIGMOID sig_cfg;
        activFunc.cfg.sigmoid = sig_cfg;
        m_config.layerActivFunc.push_back(activFunc);
    }


    // Init weights and biases
    m_config.initFunc.ptr(m_context);
}

void Model::propergate()
{
    // When not training, dont need to cache weightedSum
    // m_layers[i].output = activFunc(matrixAdd(matrixMulti(input, m_layers[i].weights), m_layers[i].bias));

    // Propergate input layer
//    m_state.layers[0].weightedSum = (m_state.layers[0].weights * *m_state.input) + m_state.layers[0].bias;
    m_state.layers[0].weightedSum = (m_state.layers[0].weights * m_state.input) + m_state.layers[0].bias;
    m_state.layers[0].output = Matrix::apply(m_state.layers[0].weightedSum, m_config.layerActivFunc[0].ptr, m_context);

    for (unsigned i = 1; i < m_state.layers.size(); ++i) {
        // Propergate hidden and output layers
        m_state.layers[i].weightedSum = (m_state.layers[i].weights * m_state.layers[i - 1].output) + m_state.layers[i].bias;
        m_state.layers[i].output = Matrix::apply(m_state.layers[i].weightedSum, m_config.layerActivFunc[i].ptr, m_context);
    }
}

const Vector& Model::propergate(const Vector& input)
{
    if (input.size() != m_config.sizeVec[0]) {
        THROW_ERROR("Invalid input size: "
                    << input.size()
                    << ", Expected: "
                    << m_config.sizeVec[0]);
    }

    m_state.input = &input;

    propergate();

    return output();
}

void Model::train(const Matrix& input, const Matrix& target)
{
    // Check sizes
    if (input.rows() != target.rows()) {
        THROW_ERROR("Input and Target matricies must have equal number of rows.\n"
                    << input.rows()
                    << " != "
                    << target.rows());
    }

    if (input.cols() != m_config.sizeVec.front()) {
        THROW_ERROR("Number of cols in input matrix must equal number of input neurons.\n"
                    << input.cols()
                    << " != "
                    << m_config.sizeVec.front());
    }

    if (target.cols() != m_config.sizeVec.back()) {
        THROW_ERROR("Number of cols in target matrix must equal number of output neurons.\n"
                    << input.cols()
                    << " != "
                    << m_config.sizeVec.front());
    }

    // Number of training samples
    size_t nSamples = input.rows();
    unsigned nEpochs = 2;

    // Number of epochs
    for (size_t epoch = 0; epoch < nEpochs; ++epoch) {


        // Change print interval at last epoch
        if (epoch == (nEpochs - 1)) {
            m_config.printInterval = 500;
        }


        // Loop training set
        size_t inputIdx = 0;
        while (inputIdx < nSamples) {


            // Training batch
            size_t batchMax = std::min(inputIdx + m_config.batchSize, nSamples);

            // Reset averages
            m_state.avg_error.fill(0.0);
            m_state.avg_input.fill(0.0);

            // Batch
            for (size_t batchIdx = inputIdx; batchIdx < batchMax; ++batchIdx) {

                // Propergate input vector                
//                propergate(input.row(batchIdx));

                m_state.input = input.row(batchIdx);
                propergate();


                // Calc error vector, average over batch
                m_state.error = Matrix::zip(m_state.layers.back().output, target.row(batchIdx), m_config.costFunc.ptr, m_context);

                // Save error to get average
                // Note: avg_error.size() == output.size() == target.size()
                m_state.avg_error += m_state.error;

                // Average input
                m_state.avg_input += input.row(batchIdx);

                // Print
                if ((batchIdx % m_config.printInterval) == 0) {
                    printState(input.row(batchIdx), target.row(batchIdx), m_state.error, batchIdx);
                }
            }


            // Divide to get avgerages
            size_t nSamples = batchMax - inputIdx;
            m_state.avg_error /= nSamples;
            m_state.avg_input /= nSamples;


            // Optimize on avg_error
            m_config.optFunc.ptr(m_context);


            // Update index
            inputIdx += m_config.batchSize;
        }
    }
}

void Model::train(const Matrix& input, const Matrix& target, unsigned nEpochs)
{
    // Check sizes
    if (input.rows() != target.rows()) {
        THROW_ERROR("Input and Target matricies must have equal number of rows.\n"
                    << input.rows()
                    << " != "
                    << target.rows());
    }

    if (input.cols() != m_config.sizeVec.front()) {
        THROW_ERROR("Number of cols in input matrix must equal number of input neurons.\n"
                    << input.cols()
                    << " != "
                    << m_config.sizeVec.front());
    }

    if (target.cols() != m_config.sizeVec.back()) {
        THROW_ERROR("Number of cols in target matrix must equal number of output neurons.\n"
                    << input.cols()
                    << " != "
                    << m_config.sizeVec.front());
    }

    // Number of training samples
    size_t nSamples = input.rows();

    // Reset step
    m_state.step = 0;

    // Number of epochs
    for (size_t epoch = 0; epoch < nEpochs; ++epoch) {


        // Change print interval at last epoch
        if (epoch >= (nEpochs - 1)) {
            m_config.printInterval = 500;
        }

        for (unsigned i = 0; i < nSamples; ++i) {
            m_state.input = input.row(i);
            m_state.target = target.row(i);

            step();
        }
    }
}

//void Model::softMax(Vector& output)
//{
//    real sum = 0;

//    for (size_t i = 0; i < output.size(); ++i) {
//        output(i) = std::exp(output(i));
//        sum += output(i);
//    }

//    for (size_t i = 0; i < output.size(); ++i) {
//        output(i) /= sum;
//    }
//}

const Vector& Model::output()
{
    return m_state.layers.back().output;
}

void Model::save(std::string dir)
{
    (void)dir;
    // Need to save m_sizeVec, weights and biases
    // Maybe also all other config

    std::stringstream netStr;

    // Network size
    netStr << "size:\n";

    for (unsigned i = 0; i < m_config.sizeVec.size(); ++i) {
        netStr << m_config.sizeVec[i] << "\n";
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

void Model::printState(Vector input, Vector target, Vector error, size_t batchIdx)
{
    (void)input;

    // Create stream
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);



    // Index
    ss << "idx: [ " << batchIdx << " ]\n";



    // Learning rate
    ss << "lrt: [ " << std::setprecision(4) << m_config.optFunc.cfg.backprop.learningRate << std::setprecision(2) << " ]\n";



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

void Model::setInput(const Vector& input)
{
    if (input.size() != m_config.sizeVec.front()) {
        THROW_ERROR("Invalid input size: "
                    << input.size()
                    << ", Expected: "
                    << m_config.sizeVec.front());
    }

    m_state.input = &input;
}

void Model::setTarget(const Vector& target)
{
    if (target.size() != m_config.sizeVec.back()) {
        THROW_ERROR("Invalid target size: "
                    << target.size()
                    << ", Expected: "
                    << m_config.sizeVec.back());
    }

    m_state.target = &target;
}

void Model::step()
{
    // Step
    ++m_state.step;

    // Propergate
    propergate();

    // Optimize if a batch is done
    if (m_config.batchSize == 1) {

        // Calc error
        m_state.error = Matrix::zip(m_state.layers.back().output, m_state.target,
                                    m_config.costFunc.ptr, m_context);

        m_state.avg_input = m_state.input;

        // Optimize
        m_config.optFunc.ptr(m_context);

    } else {

        // Save error and input to get average later
        m_state.error += Matrix::zip(m_state.layers.back().output, m_state.target,
                                         m_config.costFunc.ptr, m_context);

        m_state.avg_input += m_state.input;

        // Optimize if a batch is done
        if ((m_state.step % m_config.batchSize) == 0) {

            // Averages
            m_state.error /= m_config.batchSize;
            m_state.avg_input /= m_config.batchSize;

            // Set input to optimize on
//            m_state.input = &m_state.avg_input;

            // Optimize
            m_config.optFunc.ptr(m_context);

            // Reset averages
            m_state.error.fill(0.0);
            m_state.avg_input.fill(0.0);
        }
    }

    // Print
    if ((m_state.step % m_config.printInterval) == 1) {
        printState(m_state.avg_input, m_state.target, m_state.error, m_state.step);
    }
}

Config &Model::config()
{
    return m_config;
}

// Initialization functions
void Model::setInitializationFunction(InitializationFunction::ALL_ZERO init_func)
{
    m_config.initFunc.cfg.all_zero = init_func;
    m_config.initFunc.type = Config::InitFuncType::ALL_ZERO;
    m_config.initFunc.ptr = init_func_random_normal;
}

void Model::setInitializationFunction(InitializationFunction::RANDOM_NORMAL init_func)
{
    m_config.initFunc.cfg.random = init_func;
    m_config.initFunc.type = Config::InitFuncType::RANDOM;
    m_config.initFunc.ptr = init_func_random_normal;
}

void Model::setInitializationFunction(InitializationFunction::RANDOM_UNIFORM init_func)
{
    m_config.initFunc.cfg.uniform = init_func;
    m_config.initFunc.type = Config::InitFuncType::UNIFORM;
    m_config.initFunc.ptr = init_func_random_uniform;
}

void Model::setInitializationFunction(void (*initFunc)(Context&))
{
    m_config.initFunc.type = Config::InitFuncType::CUSTOM;
    m_config.initFunc.ptr = initFunc;
}

// Activation functions
void Model::setActivationFunction(unsigned int layerIdx, ActivationFunction::RELU activ_func)
{
    m_config.layerActivFunc[layerIdx].cfg.relu = activ_func;
    m_config.layerActivFunc[layerIdx].type = Config::ActivFuncType::RELU;
    m_config.layerActivFunc[layerIdx].ptr = activation_func_relu;
    m_config.layerActivFunc[layerIdx].derivPtr = activation_func_relu_deriv;
}

void Model::setActivationFunction(unsigned int layerIdx, ActivationFunction::SIGMOID activ_func)
{
    m_config.layerActivFunc[layerIdx].cfg.sigmoid = activ_func;
    m_config.layerActivFunc[layerIdx].type = Config::ActivFuncType::SIGMOID;
    m_config.layerActivFunc[layerIdx].ptr = activation_func_sigmoid;
    m_config.layerActivFunc[layerIdx].derivPtr = activation_func_sigmoid_deriv;
}

void Model::setActivationFunction(unsigned int layerIdx, ActivationFunction::TANH activ_func)
{
    m_config.layerActivFunc[layerIdx].cfg.tanh = activ_func;
    m_config.layerActivFunc[layerIdx].type = Config::ActivFuncType::TANH;
    m_config.layerActivFunc[layerIdx].ptr = activation_func_tanh;
    m_config.layerActivFunc[layerIdx].derivPtr = activation_func_tanh_deriv;
}

void Model::setActivationFunction(unsigned int layerIdx,
                                      real (*activFunc)(real, Context&),
                                      real (*activFuncDeriv)(real, Context&))
{
    m_config.layerActivFunc[layerIdx].type = Config::ActivFuncType::CUSTOM;
    m_config.layerActivFunc[layerIdx].ptr = activFunc;
    m_config.layerActivFunc[layerIdx].derivPtr = activFuncDeriv;
}

// Cost functions
void Model::setCostFunction(real (*costFunc)(real output, real target, Context& net))
{
    m_config.costFunc.type = Config::CostFuncType::CUSTOM;
    m_config.costFunc.ptr = costFunc;
}

void Model::setCostFunction(CostFunction::DIFFERENCE cost_func)
{
    m_config.costFunc.cfg.diff = cost_func;
    m_config.costFunc.type = Config::CostFuncType::DIFFERENCE;
    m_config.costFunc.ptr = cost_func_difference;
}

void Model::setCostFunction(CostFunction::SQUARE_DIFFERENCE cost_func)
{
    m_config.costFunc.cfg.sq_diff = cost_func;
    m_config.costFunc.type = Config::CostFuncType::SQUARE_DIFFERENCE;
    m_config.costFunc.ptr = cost_func_square_difference;
}

void Model::setCostFunction(CostFunction::CROSS_ENTROPY cost_func)
{
    m_config.costFunc.cfg.x_ntrp = cost_func;
    m_config.costFunc.type = Config::CostFuncType::CROSS_ENTROPY;
    m_config.costFunc.ptr = cost_func_cross_entropy;
}

// Optimize functions
void Model::setOptimizeFunction(OptimizeFunction::TEST opt_func)
{
    m_config.optFunc.cfg.test = opt_func;
    m_config.optFunc.type = Config::OptFuncType::TEST;
    m_config.optFunc.ptr = optimize_func_backprop;
}

void Model::setOptimizeFunction(OptimizeFunction::BACKPROP opt_func)
{
    m_config.optFunc.cfg.backprop = opt_func;
    m_config.optFunc.type = Config::OptFuncType::BACKPROP;
    m_config.optFunc.ptr = optimize_func_backprop;
}

void Model::setOptimizeFunction(void (*optFunc)(Context&))
{
    m_config.optFunc.type = Config::OptFuncType::CUSTOM;
    m_config.optFunc.ptr = optFunc;
}


































































