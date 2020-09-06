#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Common/types.h>
#include <NeuralNet/matrix.h>
#include <NeuralNet/vector.h>

// Need
//  matrixMulti(real_vec, real_vec)
//  matrixScale(real, real_vec)
//  matrixElemWiseMulti(real_vec, real_vec)

struct Layer
{
    Layer(size_t prevLayerSize, size_t size) :
        output(size, 0.0),
        bias(size, 0.0),
        gradient(size, 0.0),
        weightedSum(size, 0.0),
        weights(size, prevLayerSize, 0.0),
        m_size(size)
    {

    }

    Vector output;
    Vector bias;
    Vector gradient;
    Vector weightedSum;
    Matrix weights;

    size_t size()
    {
        return m_size;
    }

private:
    size_t m_size;

};

class NeuralNet;

struct InitializationFunction
{
    enum Type
    {
        ALL_ZERO = 1,
        RANDOM
    };

    // Custom function has type = 0
    unsigned int type;
    void (*ptr)(NeuralNet*);
};

struct CostFunction
{
    enum Type
    {
        DIFFERENCE = 1,
        CROSS_ENTROPY,
        SQUARE_DIFFERENCE
    };

    // Custom function has type = 0
    unsigned int type;
    Vector (*ptr)(const Vector&, const Vector&, NeuralNet*);
};

struct ActivationFunction
{
    enum Type
    {
        RELU = 1,
        SIGMOID,
        TANH
    };

    // Custom function has type = 0
    unsigned int type;
    Vector (*ptr)(const Vector&, NeuralNet*);
    Vector (*derivPtr)(const Vector&, NeuralNet*);
};

struct OptimizeFunction
{
    enum Type
    {
        TEST = 1,
        BACKPROP
    };

    // Custom function has type = 0
    unsigned int type;
    void (*ptr)(const Vector&, const Vector&, NeuralNet*);
};

class NeuralNet
{
public:
    NeuralNet(const std::initializer_list<size_t>& list);
    NeuralNet(const std::vector<size_t>& vec);

    // Need functions to continuesly monitor status during training
    // Graphs and stuff
    // GUI app will be another project, more clean and it will use Qt anyway

    // Function to save network

    void propergate(const Vector& input);
//    real train(real_matrix input, real_matrix target);
    void train(const Matrix& input, const Matrix& target);
    real test(const Matrix& input, const Matrix& target);
//    void backpropergate(real_vec input, real_vec error);
//    void softMax(real_vec& vec);
    void softMax(Vector& vec);
    const Vector& output();
    void printState(Vector input, Vector target, Vector error, size_t batchIdx);

    std::vector<size_t> sizeVec(); // Should be const, I guess
    std::vector<Layer>& layers(); // Should be const, I guess
    unsigned int batchSize();
    void setBatchSize(unsigned int batchSize);
    unsigned int nEpochs();
    void setNEpochs(unsigned int nEpochs);
    unsigned int printInterval();
    void setPrintInterval(unsigned int printInterval);
    void softMax();
    void setSoftMax(bool softMax);
    real learningRate();
    void setLearningRate(real learningRate);


    // Might switch enums to structs
    // Like so:
    // InitFunc::Random rnd = {1, 0.3, false};
    // setInitializationFunction(rnd);
    // And have a seperate function for each built-in type
    // Custom functions will work as is
    void setInitializationFunction(InitializationFunction::Type init_func);
    void setInitializationFunction(void (*initFunc)(NeuralNet*));

    void setCostFunction(CostFunction::Type cost_func);
    void setCostFunction(Vector (*costFunc)(const Vector&, const Vector&, NeuralNet*));

    Vector activationFunction(unsigned int layerIdx, const Vector& x);
    Vector activationFunctionDerivate(unsigned int layerIdx, const Vector& x);

    void setHiddenLayerActivationFunction(ActivationFunction::Type activ_func);
    void setHiddenLayerActivationFunction(Vector (*activFunc)(const Vector&, NeuralNet*), Vector (*activFuncDeriv)(const Vector&, NeuralNet*));
    void setOutputLayerActivationFunction(ActivationFunction::Type activ_func);
    void setOutputLayerActivationFunction(Vector (*activFunc)(const Vector&, NeuralNet*), Vector (*activFuncDeriv)(const Vector&, NeuralNet*));

    void setActivationFunction(unsigned int layerIdx, ActivationFunction::Type activ_func);
    void setActivationFunction(unsigned int layerIdx, Vector (*activFunc)(const Vector&, NeuralNet*), Vector (*activFuncDeriv)(const Vector&, NeuralNet*));

    void setOptimizeFunction(OptimizeFunction::Type opt_func);
    void setOptimizeFunction(void (*optFunc)(const Vector&, const Vector&, NeuralNet*));

private:
    std::vector<size_t> m_sizeVec;
    std::vector<Layer> m_layers;
    unsigned int m_batchSize;
    unsigned int m_nEpochs;
    unsigned int m_printInterval;
    bool m_softMax;
    real m_learningRate;

    InitializationFunction m_initFunc;
    CostFunction m_costFunc;
    std::vector<ActivationFunction> m_layerActivFunc;
    OptimizeFunction m_optFunc;

};

#endif // NEURAL_NET_H

