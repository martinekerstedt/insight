#ifndef STATE_H
#define STATE_H

#include <Matrix/vector.h>
#include <Matrix/vectorview.h>

namespace NeuralNet
{

struct State
{
    struct Layer
    {
        Layer(size_t prevLayerSize, size_t size) :
            output(size, 0.0),
            bias(size, 0.0),
            gradient(size, 0.0),
            weightedSum(size, 0.0),
            weights(size, prevLayerSize, 0.0),
            m_size(size) {}
        Vector output;
        Vector bias;
        Vector gradient;
        Vector weightedSum;
        Matrix weights;
        size_t m_size;

        size_t size()
        {
            return m_size;
        }
    };

    State(const std::vector<size_t>& sizeVec) :
        input(sizeVec[0]),
        avg_input(sizeVec[0], 0.0),
        target(sizeVec.back()),
        error(sizeVec.back(), 0.0),
        avg_error(sizeVec.back(), 0.0),
        step(0),
        learningRate(0.0)
    {
        for (size_t i = 0; i < (sizeVec.size() - 1); ++i) {
            layers.push_back(Layer(sizeVec[i], sizeVec[i + 1]));
        }
    }

    // Input
    VectorView input;
    Vector avg_input;

    // Target
    VectorView target;

    // Error
    Vector error;
    Vector avg_error;

    // Step
    unsigned step;

    // Learning rate
    real learningRate;

    // Layers
    std::vector<Layer> layers;

};

//struct State
//{
//    // Precepton layer
//    struct Layer
//    {
//        Layer(size_t prevLayerSize, size_t size) :
//            output(size, 0.0),
//            bias(size, 0.0),
//            gradient(size, 0.0),
//            weightedSum(size, 0.0),
//            weights(size, prevLayerSize, 0.0),
//            m_size(size)
//        {

//        }

//        Vector output;
//        Vector bias;
//        Vector gradient;
//        Vector weightedSum;
//        Matrix weights;

//        size_t size()
//        {
//            return m_size;
//        }

//    private:
//        size_t m_size;

//    };

//    std::vector<Layer> layers;

//    // Input
//    VectorView input;
//    Vector avg_input;

//    // Target
//    VectorView target;

//    // Error
//    Vector error;
//    Vector avg_error;

//    // Step
//    unsigned step;

//    // Learning rate
//    real learningRate;

//};


} // namespace NeuralNet


#endif // STATE_H































































