#include <NeuralNet/neuralnet.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>

#include <Common/mnist.h>

#include <random>




//struct SomeFunction
//{

//};

namespace SomeFunction
{

struct FuncOne
{
    int a;
    bool b;
};

struct FuncTwo
{
    int c;
    bool d;
};

}


void my_init(NeuralNet::Context& net)
{
    std::vector<NeuralNet::State::Layer>& layers = net.layers;
    layers[0].bias(2) = 2;

//    SomeFunction::FuncTwo f1;
}


int main()
{
    // Load data
    MNIST mnist;
    Matrix train_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/train-labels.idx1-ubyte");
    Matrix train_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/train-images.idx3-ubyte");
    Matrix test_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-labels.idx1-ubyte");
    Matrix test_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-images.idx3-ubyte");
//    Matrix train_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/fashion-mnist/train-labels.idx1-ubyte");
//    Matrix train_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/fashion-mnist/train-images.idx3-ubyte");
//    Matrix test_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/fashion-mnist/t10k-labels.idx1-ubyte");
//    Matrix test_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/fashion-mnist/t10k-images.idx3-ubyte");


    // Config net
    NeuralNet::Model net({784, 32, 10});
//    NeuralNet net({784, 10, 10});
//    NeuralNet net({784, 64, 32, 10});

    net.config().batchSize = 1;
    net.config().printInterval = 10000;


//    OptimizeFunction::BACKPROP cfg;
//    cfg.learningRate = 0.08;
//    net.setOptimizeFunction(cfg);

//    LearningRateFunction::CONSTANT cfg;
//    cfg.initial = 0.01;
//    net.setLearningRateFunction(cfg);

//    LearningRateFunction::EXPONETIAL_DECAY cfg;
//    cfg.initial = 0.1;
//    cfg.decay_rate = 0.96;
//    cfg.decay_steps = 10000;
//    cfg.step = 0;
//    net.setLearningRateFunction(cfg);

//    LearningRateFunction::LINEAR_DECAY cfg;
//    net.setLearningRateFunction(cfg);

//    InitializationFunction::RANDOM_UNIFORM cfg2;
//    net.setInitializationFunction(cfg2);
        
    
    // Train
    net.train(train_images, train_labels, 2);


    // Test
    unsigned nSamples = test_labels.rows();
    real correctPrecentage = nSamples;

    for (unsigned i = 0; i < nSamples; ++i) {

        // Propergate
        net.propergate(test_images.row(i));


        // Check
        real res = 0.0;
        real mean = 0.0;
        unsigned resIdx = 0;
        bool uncertainRes = false;
        const Vector& output = net.output();

        for (unsigned j = 0; j < output.size(); ++j) {
            // Find max value
            if (res < output(j)) {
                res = output(j);
                resIdx = j;
            }

            // Sum all
            mean += output(j);
        }

        // Calc mean
        mean /= output.size();


        // Check if res is correct
        for (unsigned j = 0; j < 10; ++j) {
            if (test_labels(i, j) == 1) {
                if (j != resIdx) {
                    // Uncertain res, well just wrong!
                    uncertainRes = true;
                    --correctPrecentage;
                    break;
                }
            }
        }


        // Check if max value is close to mean
        if (std::abs(mean - res) >= 0.1) {

            // Uncertain res
            uncertainRes = true;

        } else {

            // Check for values close to max value
            for (unsigned j = 0; j < output.size(); ++j) {
                if (output(j) > (res - 0.05)) {
                    // Uncertain res
                    uncertainRes = true;
                    break;
                }
            }
        }


        // Print
        if (uncertainRes) {
//            continue;
        }


    }

    // Model accuarcy
    correctPrecentage /= nSamples;

    std::cout << "Accuracy: " << correctPrecentage << std::endl;


    return 0;
}











































