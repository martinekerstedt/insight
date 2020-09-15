#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <NeuralNet/neuralnet.h>

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

#endif // FUNCTIONS_H
