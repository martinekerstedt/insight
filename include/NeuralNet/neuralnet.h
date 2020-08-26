#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Common/types.h>

// Need
//  matrixMulti(real_vec, real_vec)
//  matrixScale(real, real_vec)
//  matrixElemWiseMulti(real_vec, real_vec)

struct Neuron
{
    Neuron(size_t size)
    {
        weights.resize(size, 0);
    }

    real_vec weights;   
    real bias = 0;
    real output = 0;
    real gradient = 0;
    real weightedSum = 0;
//    real weights[size];

    size_t size()
    {
        return weights.size();
    }

    real& operator [](int i)
    {
        return weights[i];
    }

    real operator [](int i) const
    {
        return weights[i];
    }
};

typedef std::vector<Neuron> Layer;

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
    real (*ptr)(real, real, NeuralNet*);
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
    real (*ptr)(real, NeuralNet*);
    real (*derivPtr)(real, NeuralNet*);
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
    void (*ptr)(real_vec, real_vec, NeuralNet*);
};

class NeuralNet
{
public:
    NeuralNet(std::vector<size_t> size);    

    void propergate(real_vec input);
    real train(real_matrix input, real_matrix target);    
//    void backpropergate(real_vec input, real_vec error);
    void softMax(real_vec& vec);
    void softMax(Layer& vec);
    void printState(real_vec input, real_vec target, real_vec error, size_t batchIdx);

    std::vector<size_t> sizeVec();
    std::vector<Layer>& layers();
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
    void setCostFunction(real (*costFunc)(real, real, NeuralNet*));

    real activationFunction(unsigned int layerIdx, real x);
    real activationFunctionDerivate(unsigned int layerIdx, real x);
    void setHiddenLayerActivationFunction(ActivationFunction::Type activ_func);
    void setHiddenLayerActivationFunction(real (*activFunc)(real, NeuralNet*), real (*activFuncDeriv)(real, NeuralNet*));
    void setOutputLayerActivationFunction(ActivationFunction::Type activ_func);
    void setOutputLayerActivationFunction(real (*activFunc)(real, NeuralNet*), real (*activFuncDeriv)(real, NeuralNet*));
    void setActivationFunction(unsigned int layerIdx, ActivationFunction::Type activ_func);
    void setActivationFunction(unsigned int layerIdx, real (*activFunc)(real, NeuralNet*), real (*activFuncDeriv)(real, NeuralNet*));

    void setOptimizeFunction(OptimizeFunction::Type opt_func);
    void setOptimizeFunction(void (*optFunc)(real_vec, real_vec, NeuralNet*));

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
