#ifndef STATE_H
#define STATE_H

#include <Matrix/vector.h>
#include <Matrix/vectorview.h>

namespace NeuralNet
{


struct State
{
    // Precepton layer
    struct Layer
    {
        Layer(size_t prevLayerSize, size_t size) :
            output(size, 0.0),
            bias(size, 0.0),
            gradient(size, 0.0),
            weightedSum(size, 0.0),
            weights(size, prevLayerSize, 0.0),
            m_size(size)
        {

        }

        Vector output;
        Vector bias;
        Vector gradient;
        Vector weightedSum;
        Matrix weights;

        size_t size()
        {
            return m_size;
        }

    private:
        size_t m_size;

    };

    std::vector<Layer> layers;

    // Input
    VectorView input;
    Vector avg_input;

    // Target
    VectorView target;

    // Error
    Vector error;
    Vector avg_error;

    // Step
    unsigned long step;

};


} // namespace NeuralNet


#endif // STATE_H































































