#include "NeuralNet/neuralnet.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>

#include <Common/mnist.h>

//class Matrix2
//{
//public:
//    Matrix2()
//    {

//    }

//    real& operator [](int i)
//    {
//        return m_data[i];
//    }

//    real operator [](int i) const
//    {
//        return m_data[i];
//    }

//private:
//    unsigned int m_height;
//    unsigned int m_width;
//    std::vector<real> m_data;
//};

//struct Layer2
//{
//    real_vec output;
//    real_vec bias;
//    real_vec gradient;
//    real_vec weightedSum;
//    real_matrix weights;

//    size_t size()
//    {
//        return 8;
//    }
//};

//std::vector<Layer2> network;

//real matrixMulti(real_matrix a, real_matrix b)
//{
//    return 2;
//}

//real matrixMulti(real_vec a, real_vec b)
//{
//    return 2;
//}

//real_vec matrixMulti(real_vec a, real_matrix b)
//{
//    return real_vec();
//}

//real matrixMulti(real_matrix a, real_vec b)
//{
//    return 2;
//}

//real_vec matrixAdd(real_vec a, real_vec b)
//{
//    return real_vec();
//}

//real_vec matrixScale(real a, real_vec b)
//{
//    return real_vec();
//}

//real matrixElemWiseMulti(real_vec a, real_vec b)
//{
//    return 2;
//}

//real_vec activFunc(real_vec x)
//{
//    return x;
//}

//real activFuncDeriv(real x)
//{
//    return x;
//}

//real_vec propergateNetwork(real_vec input)
//{
//    // Propergate input layer
//    network[0].weightedSum = matrixAdd(matrixMulti(input, network[0].weights), network[0].bias);
//    network[0].output = activFunc(network[0].weightedSum);

//    // When not training, dont need to cache weightedSum
//    network[0].output = activFunc(matrixAdd(matrixMulti(input, network[0].weights), network[0].bias));


//    for (unsigned int i = 1; i < network.size(); ++i) {
//        // Propergate hidden and output layers
//        network[i].weightedSum = matrixAdd(matrixMulti(network[i - 1].output, network[i].weights), network[i].bias);
//        network[i].output = activFunc(network[i].weightedSum);

//        // When not training, dont need to cache weightedSum
//        network[i].output = activFunc(matrixAdd(matrixMulti(input, network[i].weights), network[i].bias));
//    }

//    return network.back().output;
//}

//void backpropNetwork(real_vec error)
//{
//    real learningRate = 0.01;

//    // Backprop output layer
//    network.back().gradient = matrixScale(learningRate, error);

//    // Loop backwards
//    for (int i = (network.back().size() - 2); i >= 0; --i) {

//        // Backprop hidden and input layers
//        network[i].gradient =
//    }
//}

#include <random>

void test_matrix()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<real> d{0, 10};


//    Matrix mat1(3, 3);
//    for (unsigned i = 0; i < mat1.rows(); ++i) {
//        for (unsigned j = 0; j < mat1.cols(); ++j) {
//            mat1(i, j) = d(gen);
//        }
//    }

//    std::cout << "mat1:\n" << mat1.str() << std::endl;



//    Matrix mat2(3, 3);
//    for (unsigned i = 0; i < mat2.rows(); ++i) {
//        for (unsigned j = 0; j < mat2.cols(); ++j) {
//            mat2(i, j) = d(gen);
//        }
//    }

//    std::cout << "mat2:\n" << mat2.str() << std::endl;


//    Matrix mat3 = mat1.transpose();

//    std::cout << "mat3:\n" << mat3.str() << std::endl;








//    Matrix mat4(3, 3);
//    for (unsigned i = 0; i < mat4.getRows(); ++i) {
//        for (unsigned j = 0; j < mat4.getCols(); ++j) {
//            mat4(i, j) = d(gen);
//        }
//    }

//    std::cout << mat4.str() << std::endl;


//    Matrix mat5(1, 1);
//    for (unsigned i = 0; i < mat5.getRows(); ++i) {
//        for (unsigned j = 0; j < mat5.getCols(); ++j) {
//            mat5(i, j) = d(gen);
//        }
//    }

//    std::cout << mat5.str() << std::endl;



//    Matrix mat1(5, 1);
//    std::cout << mat1.str() << std::endl;

//    Matrix mat2(1, 5);
//    std::cout << mat2.str() << std::endl;

//    Matrix mat3(5, 5);
//    std::cout << mat3.str() << std::endl;

}

//namespace op {
//std::pair<int, int> divmod(int x, int y) {
//    return { x / y, x % y };
//}
//}

//void test_op()
//{
//    auto divmod = base::make_named_operator(op::divmod);
//    int x = 42;
//    int y = 23;
//    auto z = x <divmod> y;
//    std::cout << "(" << z.first << ", "
//              << z.second << ")\n";
//}

int main()
{
//    test_matrix();
//    return 0;

//    test_op();
//    return 0;

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
//    net.setHiddenLayerActivationFunction(ActivationFunction::SIGMOID);
//    net.setOutputLayerActivationFunction(ActivationFunction::SIGMOID);
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











































