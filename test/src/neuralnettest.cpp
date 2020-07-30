#include "gtest/gtest.h"
#include "NeuralNet/neuralnet.h"


TEST(NeuralNetConstruct, SingleLayerSingleNeuron)
{

    std::vector<size_t> size_vec = {1, 1};
    NeuralNet net(size_vec);

    EXPECT_EQ(net.layers.size(), 1);
    EXPECT_EQ(net.layers.back().size, 1);
    EXPECT_EQ(net.layers.back().neurons.size(), 1);
}

TEST(NeuralNetConstruct, SingleLayerMultipleNeurons)
{

    std::vector<size_t> size_vec = {1, 50};
    NeuralNet net(size_vec);

    EXPECT_EQ(net.layers.size(), 1);
    EXPECT_EQ(net.layers.back().size, 50);

    for (size_t j = 0; j < net.layers[0].size; ++j) {
        EXPECT_EQ(net.layers[0][j].size, 1);
    }
}

TEST(NeuralNetConstruct, MultipleLayersSingleNeuron)
{

    std::vector<size_t> size_vec = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    NeuralNet net(size_vec);

    EXPECT_EQ(net.layers.size(), size_vec.size() - 1);

    for (size_t i = 0; i < net.layers.size(); ++i) {        
        EXPECT_EQ(net.layers[i].size, 1);
        EXPECT_EQ(net.layers[i][0].size, 1);
    }
}

TEST(NeuralNetConstruct, MultipleLayersMultipleNeurons)
{

    std::vector<size_t> size_vec = {1, 20, 51, 45, 99, 2, 77, 23, 57, 1, 34, 71, 3};
    NeuralNet net(size_vec);

    EXPECT_EQ(net.layers.size(), size_vec.size() - 1);

    for (size_t i = 0; i < net.layers.size(); ++i)  {

        EXPECT_EQ(net.layers[i].size, size_vec[i + 1]);

        for (size_t j = 0; j < net.layers[i].size; ++j) {
            EXPECT_EQ(net.layers[i][j].size, size_vec[i]);
        }
    }
}

TEST(NeuralNetTrain, XOR)
{
    std::vector<size_t> size_vec = {2, 6, 2, 1};
    NeuralNet net(size_vec);

    size_t nLoops = 10000;
    
    for (size_t i = 0; i < nLoops; ++i)
    {
        real_vec input;

        if (rand() < (RAND_MAX/2))
        {
            input.push_back(0);
        } else {
            input.push_back(1);
        }
        
        if (rand() < (RAND_MAX/2))
        {
            input.push_back(0);
        } else {
            input.push_back(1);
        }

        real_vec target;
        
        target.push_back(static_cast<int>(input[0])^static_cast<int>(input[1]));
        



        if (i < (nLoops - 50))
        {
            // real_vec error = net.train(input, target);

            // std::cout << "in1: " << input[0]
            //             << "\tin2: " << input[1]
            //             << "\ttar: " << target[0]
            //             << "\ter: " << error[0] << std::endl;

        } else {

            // Last 50
            net.propergate(input);

            std::cout << "in1: " << input[0]
                    << "\tin2: " << input[1]
                    << "\ttar: " << target[0]
                    << "\tout: " << net.layers.back().neurons.back().output << std::endl;
        }
    }
    
}
