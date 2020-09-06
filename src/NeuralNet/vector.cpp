#include <NeuralNet/vector.h>

Vector::Vector() :
    Matrix()
{

}

Vector::Vector(unsigned size) :
    Matrix(size, 1)
{

}

Vector::Vector(unsigned size, const real& initval) :
    Matrix(size, 1, initval)
{

}

Vector::Vector(const Matrix& mat) :
    Matrix(mat.size(), 1, mat.vec())
{

}

Vector::Vector(const std::vector<real>& vec) :
    Matrix(vec)
{

}

Vector::Vector(const std::initializer_list<real>& list) :
    Matrix(list)
{

}

Vector Vector::transpose() const
{
    Vector res(*this);
    res.m_rows = m_cols;
    res.m_cols = m_rows;
    return res;
}

void Vector::pushBack(const real &elem)
{
    m_vec.push_back(elem);

    if ((m_rows == 0) || (m_cols == 0)) {
        m_rows = 1;
        m_cols = 1;
    } else if (m_rows >= m_cols) {
        ++m_rows;
    } else {
        ++m_cols;
    }
}

void Vector::popBack()
{
    if ((m_rows == 0) || (m_cols == 0)) {
        m_rows = 0;
        m_cols = 0;
        m_vec.clear();
    } else if (m_rows >= m_cols) {
        if (m_rows == 1) {
            m_rows = 0;
            m_cols = 0;
        } else {
            --m_rows;
        }
    } else {
        --m_cols;
    }

    m_vec.pop_back();
}

real Vector::front() const
{
    return m_vec.front();
}

real Vector::back() const
{
    return m_vec.back();
}

void Vector::reserve(size_t size)
{
    m_vec.reserve(size);
}



























































