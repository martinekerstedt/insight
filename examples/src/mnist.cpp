#include "NeuralNet/neuralnet.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>

#include <Common/mnist.h>

int main()
{
    MNIST mnist;
    real_vec train_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/train-labels.idx1-ubyte");
    std::vector<real_matrix> train_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/train-images.idx3-ubyte");
    real_vec test_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-labels.idx1-ubyte");
    std::vector<real_matrix> test_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-images.idx3-ubyte");

    std::vector<size_t> size_vec = {784/* 28*28 */, 32, 10};
    NeuralNet net(size_vec);


    net.setNEpochs(3);
    net.setBatchSize(1);
    net.setLearningRate(0.01);
    net.setPrintInterval(10000);
    net.setHiddenLayerActivationFunction(ActivationFunction::SIGMOID);
    net.setOutputLayerActivationFunction(ActivationFunction::SIGMOID);
    net.setCostFunction(CostFunction::SQUARE_DIFFERENCE);
    net.setSoftMax(false);


    real_matrix input;
    real_matrix target;

    for (size_t i = 0; i < train_labels.size(); ++i) {
        // Format input
        real_vec input_sample;
        input_sample.reserve(28*28);

        for (size_t j = 0; j < train_images[i].size(); ++j) {
            for (size_t k = 0; k < train_images[i][j].size(); ++k) {
                input_sample.push_back(train_images[i][j][k]);
            }
        }

        input.push_back(input_sample);


        // Format target
        real_vec target_sample;
        for (size_t j = 0; j < 10; ++j) {

            if (j == (unsigned int)train_labels[i]) {
                target_sample.push_back(1);
            } else {
                target_sample.push_back(0);
            }
        }

        target.push_back(target_sample);
    }



    // Train
    net.train(input, target);



    // Test
    input.clear();
    target.clear();

    real correctPrecentage = test_labels.size();

    for (size_t i = 0; i < test_labels.size(); ++i) {
        // Format input
        real_vec input_sample;
        input_sample.reserve(28*28);

        for (size_t j = 0; j < test_images[i].size(); ++j) {
            for (size_t k = 0; k < test_images[i][j].size(); ++k) {
                input_sample.push_back(test_images[i][j][k]);
            }
        }

        // Propergate
        net.propergate(input_sample);


        // Format target
//        real_vec target_sample;
//        for (size_t j = 0; j < 10; ++j) {

//            if (j == (unsigned int)test_labels[i]) {
//                target_sample.push_back(1);
//            } else {
//                target_sample.push_back(0);
//            }
//        }


        // Check
        real res = 0.0;
        real mean = 0.0;
        size_t resIdx = 0;
        bool uncertainRes = false;
        std::vector<Layer> layers = net.layers();

        for (size_t j = 0; j < layers.back().size(); ++j) {
            // Find max value
            if (res < layers.back()[j].output) {
                res = layers.back()[j].output;
                resIdx = j;
            }

            // Sum all
            mean += layers.back()[j].output;
        }

        // Calc mean
        mean /= layers.back().size();


        // Check if res is correct
        if (test_labels[i] != resIdx) {
            // Uncertain res, well just wrong!
            uncertainRes = true;
            --correctPrecentage;
        }


        // Check if max value is close to mean
        if (std::abs(mean - res) >= 0.1) {

            // Uncertain res
            uncertainRes = true;

        } else {

            // Check for values close to max value
            for (size_t j = 0; j < layers.back().size(); ++j) {
                if (layers.back()[j].output > (res - 0.05)) {
                    // Uncertain res
                    uncertainRes = true;
                    break;
                }
            }
        }


        // Print
        if (!uncertainRes) {
            continue;
        }


    }

    // Model accuarcy
    correctPrecentage /= test_labels.size();

    std::cout << "Accuracy: " << correctPrecentage << std::endl;



    return 0;
}











































