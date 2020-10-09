#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Common/types.h>
#include <NeuralNet/matrix.h>
#include <NeuralNet/vector.h>


// Introduce step? Will make monitoring easier
// One step is:
//   Propegate input
//   Calc cost
//   If batchSize allows, optimize
// Need to set input in seperate function
// Should be enough to config network then call step in a loop
// until a flag is set to signal that training is done

// Want to be able to easly chain nerual nets together
//   Set input
//   Step
//   Get output

// Only for a single network
// startTraining()
// Run training:
//   Initialize
//   Loop step until input and epochs are exhuseted


// Might intorduce preprocessing and postprocessing functions
// softMax, convolutions, etc.


// Should only have to save state of network to be able to load later
// State must contain everything that is need to save/load


// Allow to have user defined functions that dont have net as an arg


class NeuralNet
{
public:
    NeuralNet();
    NeuralNet(const std::initializer_list<size_t>& list);
    NeuralNet(const std::vector<size_t>& vec);

    class StateAccess;
    struct State;

    // Need functions to continuesly monitor status during training
    // Graphs and stuff
    // GUI app will be another project, more clean and it will use Qt anyway

    // Function to save network

    void propergate(const Vector& input);
    void train(const Matrix& input, const Matrix& target);
    void softMax(Vector& vec);
    const Vector& output();
    void save(std::string dir);
    void printState(Vector input, Vector target, Vector error, size_t batchIdx);

    void setTraningData(const Matrix& input, const Matrix& target);
    void step();


    const State& state();

    void setInitializationFunction(void (*initFunc)(StateAccess&));
    void setInitializationFunction(InitializationFunction::ALL_ZERO init_func);
    void setInitializationFunction(InitializationFunction::RANDOM_NORMAL init_func);

    void setCostFunction(real (*costFunc)(real, real, StateAccess&));
    void setCostFunction(CostFunction::DIFFERENCE cost_func);
    void setCostFunction(CostFunction::CROSS_ENTROPY cost_func);
    void setCostFunction(CostFunction::SQUARE_DIFFERENCE cost_func);

    void setActivationFunction(unsigned int layerIdx,
                               real (*activFunc)(real, StateAccess&),
                               real (*activFuncDeriv)(real, StateAccess&));
    void setActivationFunction(unsigned int layerIdx, ActivationFunction::RELU activ_func);
    void setActivationFunction(unsigned int layerIdx, ActivationFunction::SIGMOID activ_func);
    void setActivationFunction(unsigned int layerIdx, ActivationFunction::TANH activ_func);

    void setOptimizeFunction(void (*optFunc)(const Vector&, const Vector&, StateAccess&));
    void setOptimizeFunction(OptimizeFunction::TEST opt_func);
    void setOptimizeFunction(OptimizeFunction::BACKPROP opt_func);


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

    struct State
    {
        enum class InitFuncType
        {
            CUSTOM,
            ALL_ZERO,
            RANDOM
        };

        enum class ActivFuncType
        {
            CUSTOM,
            RELU,
            SIGMOID,
            TANH
        };

        enum class CostFuncType
        {
            CUSTOM,
            DIFFERENCE,
            CROSS_ENTROPY,
            SQUARE_DIFFERENCE
        };

        enum class OptFuncType
        {
            CUSTOM,
            TEST,
            BACKPROP
        };

        struct InitFuncConfig
        {
            InitializationFunction::ALL_ZERO all_zero;
            InitializationFunction::RANDOM_NORMAL random;
        };

        struct ActivFuncConfig
        {
            ActivationFunction::RELU relu;
            ActivationFunction::SIGMOID sigmoid;
            ActivationFunction::TANH tanh;
        };

        struct CostFuncConfig
        {
            CostFunction::DIFFERENCE diff;
            CostFunction::SQUARE_DIFFERENCE sq_diff;
            CostFunction::CROSS_ENTROPY x_ntrp;
        };

        struct OptFuncConfig
        {
            OptimizeFunction::TEST test;
            OptimizeFunction::BACKPROP backprop;
        }; 

        struct Config
        {
            std::vector<size_t> sizeVec;
            unsigned int batchSize;
            unsigned int nEpochs;
            unsigned int printInterval;
            bool softMax;
        } config;

        std::vector<Layer> layers;

        struct InitFunc
        {
            InitFuncConfig cfg;
            InitFuncType type;
            void (*ptr)(StateAccess&);
        } initFunc;

        struct ActivFunc
        {            
            ActivFuncConfig cfg;
            ActivFuncType type;
            real (*ptr)(real, StateAccess&);
            real (*derivPtr)(real, StateAccess&);
        };

        std::vector<ActivFunc> layerActivFunc;

        struct CostFunc
        {
            CostFuncConfig cfg;
            CostFuncType type;
            real (*ptr)(real, real, StateAccess&);
        } costFunc;

        struct OptFunc
        {
            OptFuncConfig cfg;
            OptFuncType type;
            void (*ptr)(const Vector&, const Vector&, StateAccess&);
        } optFunc;


//        const Matrix& input;
//        const Matrix& target;
//        unsigned step;
    };

    State::Config& config();

    class StateAccess
    {

    public:
        StateAccess(State& state) :
            m_state(state)
        {

        }

        // Currently selected functions, read only
        State::InitFuncType initFuncType() { return m_state.initFunc.type; }
        State::ActivFuncType activFuncType(unsigned layerIdx) { return m_state.layerActivFunc[layerIdx].type; }
        State::CostFuncType costFuncType() { return m_state.costFunc.type; }
        State::OptFuncType optFuncType() { return m_state.optFunc.type; }


        // All user definable functions
        void initializeFunction()
        {
            m_state.initFunc.ptr(*this);
        }

        real activationFunction(real x, unsigned int layerIdx)
        {
            return m_state.layerActivFunc[layerIdx].ptr(x, *this);
        }

        real activationFunctionDerivate(real x, unsigned int layerIdx)
        {
            return m_state.layerActivFunc[layerIdx].derivPtr(x, *this);
        }

        auto activationFunction(unsigned int layerIdx)
        {
            return m_state.layerActivFunc[layerIdx].ptr;
        }

        auto activationFunctionDerivate(unsigned int layerIdx)
        {
            return m_state.layerActivFunc[layerIdx].derivPtr;
        }

        real costFunction(real output, real target)
        {
            return m_state.costFunc.ptr(output, target, *this);
        }

        void optimizeFunction(const Vector& input, const Vector& error)
        {
            m_state.optFunc.ptr(input, error, *this);
        }

        // All network configs, read only
        const State::Config config() { return m_state.config; }

        // Network stats, read only
        // timeSpentTraining
        // all kinds of time metrics really
        // nSteps

        // Layers
        std::vector<Layer>& layers() { return m_state.layers; }

        // Initialization function declarations
        friend void init_func_random_normal(StateAccess& net);

        // Activation function declarations
        friend Vector activation_func_relu(const Vector& x, StateAccess& net);
        friend Vector activation_func_sigmoid(const Vector& x, StateAccess& net);
        friend Vector activation_func_tanh(const Vector& x, StateAccess& net);

        // Activation function derivative declarations
        friend Vector activation_func_relu_deriv(const Vector& x, StateAccess& net);
        friend Vector activation_func_sigmoid_deriv(const Vector& x, StateAccess& net);
        friend Vector activation_func_tanh_deriv(const Vector& x, StateAccess& net);

        // Cost function declarations
        friend Vector cost_func_difference(const Vector& output, const Vector& target, StateAccess& net);
        friend Vector cost_func_square_difference(const Vector& output, const Vector& target, StateAccess& net);
        friend Vector cost_func_cross_entropy(const Vector& output, const Vector& target, StateAccess& net);

        // Optimizer function declarations
        friend void optimize_func_backprop(const Vector& input, const Vector& error, StateAccess& net);

    private:
        State& m_state;

    };

private:
    State m_state;
    StateAccess m_stateAccess;   

};

#endif // NEURAL_NET_H

