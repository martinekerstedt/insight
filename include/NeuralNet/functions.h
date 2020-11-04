#ifndef FUNCTIONS_H
#define FUNCTIONS_H

//#include <NeuralNet/neuralnet.h>
#include <NeuralNet/stateaccess.h>

// Initialization function declarations
//void init_func_random_normal(NeuralNet::StateAccess& net);

//// Activation function declarations
//Vector activation_func_relu(const Vector& x, NeuralNet::StateAccess& net);
//Vector activation_func_sigmoid(const Vector& x, NeuralNet::StateAccess& net);
//Vector activation_func_tanh(const Vector& x, NeuralNet::StateAccess& net);

//// Activation function derivative declarations
//Vector activation_func_relu_deriv(const Vector& x, NeuralNet::StateAccess& net);
////Vector activation_func_sigmoid_deriv(const Vector& x, NeuralNet::StateAccess& net);
//Vector activation_func_tanh_deriv(const Vector& x, NeuralNet::StateAccess& net);

//// Cost function declarations
//Vector cost_func_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);
//Vector cost_func_square_difference(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);
//Vector cost_func_cross_entropy(const Vector& output, const Vector& target, NeuralNet::StateAccess& net);

// Optimizer function declarations
//void optimize_func_backprop(const Vector& input, const Vector& error, NeuralNet::StateAccess& net);

namespace NeuralNet
{

// Initialization function declarations
void init_func_random_normal(StateAccess& net);
void init_func_random_uniform(StateAccess& net);

// Activation function declarations
real activation_func_relu(real x, StateAccess& net);
real activation_func_sigmoid(real x, StateAccess& net);
real activation_func_tanh(real x, StateAccess& net);

// Activation function derivative declarations
real activation_func_relu_deriv(real x, StateAccess& net);
real activation_func_sigmoid_deriv(real x, StateAccess& net);
real activation_func_tanh_deriv(real x, StateAccess& net);

// Cost function declarations
real cost_func_difference(real output, real target, StateAccess& net);
real cost_func_square_difference(real output, real target, StateAccess& net);
real cost_func_cross_entropy(real output, real target, StateAccess& net);

// Optimizer function declarations
void optimize_func_backprop(StateAccess& net);


} // namespace NeuralNet


#endif // FUNCTIONS_H
