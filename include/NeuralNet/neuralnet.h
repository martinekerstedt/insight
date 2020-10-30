#ifndef NEURAL_NET_H
#define NEURAL_NET_H

//#include <Common/types.h>
//#include <NeuralNet/matrix.h>
//#include <NeuralNet/vector.h>

//#include <NeuralNet/state.h>
#include <NeuralNet/stateaccess.h>

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


namespace NeuralNet
{


class Model
{

public:
    Model();
    Model(const std::initializer_list<size_t>& list);
    Model(const std::vector<size_t>& vec);

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

    State::Config& config();

    void setInitializationFunction(void (*initFunc)(StateAccess&));
    void setInitializationFunction(InitializationFunction::ALL_ZERO init_func);
    void setInitializationFunction(InitializationFunction::RANDOM_NORMAL init_func);
    void setInitializationFunction(InitializationFunction::RANDOM_UNIFORM init_func);

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

private:
    State m_state;
    StateAccess m_stateAccess;   

};


} // namespace NeuralNet


#endif // NEURAL_NET_H

