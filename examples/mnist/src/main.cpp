#include <iostream>



#include <vector>
#include <type_traits>

template <typename NumericType>
class Matrix
{
    static_assert(std::is_arithmetic<NumericType>::value, "NumericType must be numeric");

public:
    Matrix();
    virtual ~Matrix();

private:
    unsigned m_rows;
    unsigned m_cols;
    std::vector<NumericType> m_vec;

};



int main()
{

    std::cout << "Hello world!" << std::endl;

    return 0;
}











































