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
