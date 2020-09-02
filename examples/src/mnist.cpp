#include "NeuralNet/neuralnet.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>

#include <Common/mnist.h>

#include <random>

void test_matrix()
{
    Matrix mat1(4, 4, {1,  2,  3,  4,
                      5,  6,  7,  8,
                      9, 10, 11, 12,
                      13, 14, 15, 16});

    std::cout << mat1.str() << std::endl;

    Vector vec1({100, 101, 102, 103});
    mat1.addCol(vec1);

    std::cout << mat1.str() << std::endl << std::endl << std::endl;



    Matrix mat2(4, 1, {1,
                       2,
                       3,
                       4});

    std::cout << mat2.str() << std::endl;

    Vector vec2({100, 101, 102, 103});
    mat2.addCol(vec2);

    std::cout << mat2.str() << std::endl << std::endl << std::endl;



    Matrix mat3(1, 4, {1, 2, 3, 4});

    std::cout << mat3.str() << std::endl;

    Vector vec3(1, 100);
    mat3.addCol(vec3);

    std::cout << mat3.str() << std::endl << std::endl << std::endl;



    Matrix mat4(4, 2, {1,  2,
                       3,  4,
                       5,  6,
                       7,  8});

    std::cout << mat4.str() << std::endl;

    Vector vec4({100, 101, 102, 103});
    mat4.addCol(vec4);

    std::cout << mat4.str() << std::endl << std::endl << std::endl;



    Matrix mat5(2, 4, {1,  2,  3,  4,
                      5,  6,  7,  8});

    std::cout << mat5.str() << std::endl;

    Vector vec5(2);
    vec5(0) = 100;
    vec5(1) = 101;
    mat5.addCol(vec5);

    std::cout << mat5.str() << std::endl << std::endl << std::endl;
}

int main()
{
//    test_matrix();
//    return 0;

//    Matrix mat(4, 1, {1,
//                      2,
//                      3,
//                      4});

//    Vector vec;

//    Vector vec2({1,
//                2,
//                3,
//                4});

//    vec = mat;
//    vec = vec2;
//    vec = {1,
//           2,
//           3,
//           4};


    MNIST mnist;
//    Vector train_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/train-labels.idx1-ubyte");
    Vector train_labels_nbr = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/train-labels.idx1-ubyte");
//    std::vector<real_matrix> train_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/train-images.idx3-ubyte");
    Matrix train_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/train-images.idx3-ubyte");
    Vector test_labels = mnist.read_label_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-labels.idx1-ubyte");
//    std::vector<real_matrix> test_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-images.idx3-ubyte");
    Matrix test_images = mnist.read_image_file("/home/martin/Documents/Projects/insight/data/MNIST/t10k-images.idx3-ubyte");

    std::vector<size_t> size_vec = {784/* 28*28 */, 32, 10};
    NeuralNet net(size_vec);


    net.setNEpochs(4);
    net.setBatchSize(1);
    net.setLearningRate(0.01);
    net.setPrintInterval(10000);
//    net.setHiddenLayerActivationFunction(ActivationFunction::SIGMOID);
//    net.setOutputLayerActivationFunction(ActivationFunction::SIGMOID);
    net.setCostFunction(CostFunction::SQUARE_DIFFERENCE);
    net.setSoftMax(false);


//    real_matrix input;
//    real_matrix target;

//    for (size_t i = 0; i < train_labels.size(); ++i) {
//        // Format input
//        real_vec input_sample;
//        input_sample.reserve(28*28);

//        for (size_t j = 0; j < train_images[i].size(); ++j) {
//            for (size_t k = 0; k < train_images[i][j].size(); ++k) {
//                input_sample.push_back(train_images[i][j][k]);
//            }
//        }

//        input.push_back(input_sample);


//        // Format target
//        real_vec target_sample;
//        for (size_t j = 0; j < 10; ++j) {

//            if (j == (unsigned int)train_labels(i)) {
//                target_sample.push_back(1);
//            } else {
//                target_sample.push_back(0);
//            }
//        }

//        target.push_back(target_sample);
//    }


    // Format target
    Matrix train_labels(0, 10);

    for (size_t i = 0; i < train_labels_nbr.size(); ++i) {

        Vector target_vec;
        target_vec.reserve(10);

        for (size_t j = 0; j < 10; ++j) {
            if (j == (unsigned int)train_labels_nbr(i)) {
                target_vec.pushBack(1);
            } else {
                target_vec.pushBack(0);
            }
        }

        train_labels.addRow(target_vec);
    }


    // Train
//    net.train(input, target);
    net.train(train_images, train_labels);



//    // Test
//    input.clear();
//    target.clear();

//    real correctPrecentage = test_labels.size();

//    for (size_t i = 0; i < test_labels.size(); ++i) {
//        // Format input
//        real_vec input_sample;
//        input_sample.reserve(28*28);

//        for (size_t j = 0; j < test_images[i].size(); ++j) {
//            for (size_t k = 0; k < test_images[i][j].size(); ++k) {
//                input_sample.push_back(test_images[i][j][k]);
//            }
//        }

//        // Propergate
//        net.propergate(input_sample);


//        // Format target
//        real_vec target_sample;
//        for (size_t j = 0; j < 10; ++j) {

//            if (j == (unsigned int)test_labels(i)) {
//                target_sample.push_back(1);
//            } else {
//                target_sample.push_back(0);
//            }
//        }


//        // Check
//        real res = 0.0;
//        real mean = 0.0;
//        size_t resIdx = 0;
//        bool uncertainRes = false;
//        std::vector<Layer> layers = net.layers();

//        for (size_t j = 0; j < layers.back().size(); ++j) {
//            // Find max value
//            if (res < layers.back().output(j)) {
//                res = layers.back().output(j);
//                resIdx = j;
//            }

//            // Sum all
//            mean += layers.back().output(j);
//        }

//        // Calc mean
//        mean /= layers.back().size();


//        // Check if res is correct
//        if (test_labels(i) != resIdx) {
//            // Uncertain res, well just wrong!
//            uncertainRes = true;
//            --correctPrecentage;
//        }


//        // Check if max value is close to mean
//        if (std::abs(mean - res) >= 0.1) {

//            // Uncertain res
//            uncertainRes = true;

//        } else {

//            // Check for values close to max value
//            for (size_t j = 0; j < layers.back().size(); ++j) {
//                if (layers.back().output(j) > (res - 0.05)) {
//                    // Uncertain res
//                    uncertainRes = true;
//                    break;
//                }
//            }
//        }


//        // Print
//        if (!uncertainRes) {
//            continue;
//        }


//    }

//    // Model accuarcy
//    correctPrecentage /= test_labels.size();

//    std::cout << "Accuracy: " << correctPrecentage << std::endl;



    return 0;
}











































