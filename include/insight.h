#ifndef INSIGHT_H
#define INSIGHT_H

#include <NeuralNet/neuralnet.h>

class Insight
{
public:
    Insight();
    
    int square(int val);
    
private:
    NeuralNet m_neuralNet;

};

#endif // INSIGHT_H
