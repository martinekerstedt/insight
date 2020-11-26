#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <NeuralNet/context.h>

namespace NeuralNet
{

// Initialization function declarations
void init_func_random_normal(Context& net);
void init_func_random_uniform(Context& net);

// Activation function declarations
real activation_func_relu(real x, Context& net);
real activation_func_sigmoid(real x, Context& net);
real activation_func_tanh(real x, Context& net);

// Activation function derivative declarations
real activation_func_relu_deriv(real x, Context& net);
real activation_func_sigmoid_deriv(real x, Context& net);
real activation_func_tanh_deriv(real x, Context& net);

// Cost function declarations
real cost_func_difference(real output, real target, Context& net);
real cost_func_square_difference(real output, real target, Context& net);
real cost_func_cross_entropy(real output, real target, Context& net);

// Learning rate functions
void learnRate_func_constant(Context& net);
void learnRate_func_linear_decay(Context& net);
void learnRate_func_exponetial_decay(Context& net);

// Optimizer function declarations
void optimize_func_backprop(Context& net);


} // namespace NeuralNet


#endif // FUNCTIONS_H
