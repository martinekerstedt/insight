#ifndef VECTOR_H
#define VECTOR_H

#include <NeuralNet/matrix.h>

class Vector final : public Matrix
{
public:
    Vector();
    Vector(unsigned size);
    Vector(unsigned size, const real& initval);
    Vector(const Matrix& mat);
    Vector(const std::vector<real>& vec);
    Vector(const std::initializer_list<real>& list);

    template <Expr E>
    Vector(const E& expr) : Matrix(expr) {}

    void resize(unsigned size, real val = 0.0);

    void pushBack(const real& elem);
    void popBack();

    real front() const;
    real back() const;

    template <Expr E>
    Vector operator=(const E& rhs)
    {
        Matrix::operator=(rhs);

        return *this;
    }

    std::vector<real, std::allocator<real>>::iterator begin();
    std::vector<real, std::allocator<real>>::iterator end();
    std::vector<real, std::allocator<real>>::const_iterator cbegin() const;
    std::vector<real, std::allocator<real>>::const_iterator cend() const;

    // Deleted functions
    void addRow(const Vector& row) = delete;
    void addCol(const Vector& col) = delete;

};







































































//#include <Common/types.h>
//#include <vector>

//class Matrix;

//class Vector
//{
//public:
//    Vector();
//    Vector(unsigned size);
//    Vector(unsigned size, const real& initval);
//    Vector(const Vector& rhs);
//    Vector(const Matrix& rhs);
//    Vector(const std::vector<real>& rhs);
//    virtual ~Vector();

//    // Assignment operators
//    Vector& operator=(const Vector& rhs);
//    Vector& operator=(const Matrix& rhs);
//    Vector& operator=(const std::vector<real>& rhs);

//    // Vector/Vector operators
//    bool operator==(const Vector& rhs);
//    bool operator!=(const Vector& rhs);
//    Vector operator+(const Vector& rhs);
//    Vector operator-(const Vector& rhs);
//    Vector operator*(const Vector& rhs);
//    Vector operator/(const Vector& rhs);
//    Vector& operator+=(const Vector& rhs);
//    Vector& operator-=(const Vector& rhs);
//    Vector& operator*=(const Vector& rhs);
//    Vector& operator/=(const Vector& rhs);
//    Matrix matMul(const Vector& rhs);

//    // Vector/std::vector operators
//    bool operator==(const std::vector<real>& rhs);
//    bool operator!=(const std::vector<real>& rhs);
//    Vector operator+(const std::vector<real>& rhs);
//    Vector operator-(const std::vector<real>& rhs);
//    Vector operator*(const std::vector<real>& rhs);
//    Vector operator/(const std::vector<real>& rhs);
//    Vector& operator+=(const std::vector<real>& rhs);
//    Vector& operator-=(const std::vector<real>& rhs);
//    Vector& operator*=(const std::vector<real>& rhs);
//    Vector& operator/=(const std::vector<real>& rhs);

//    // Vector/scalar operators
//    Vector operator+(const real& rhs);
//    Vector operator-(const real& rhs);
//    Vector operator*(const real& rhs);
//    Vector operator/(const real& rhs);
//    Vector& operator+=(const real& rhs);
//    Vector& operator-=(const real& rhs);
//    Vector& operator*=(const real& rhs);
//    Vector& operator/=(const real& rhs);

//    // Vector/Matrix operators
//    bool operator==(const Matrix& rhs);
//    bool operator!=(const Matrix& rhs);
//    Vector operator*(const Matrix& rhs);
//    Vector& operator*=(const Matrix& rhs);

//    // Access
//    std::vector<real>& vec();
//    const std::vector<real>& vec() const;
//    real& operator()(const unsigned& idx);
//    const real& operator()(const unsigned& idx) const;
//    unsigned size() const;
//    real& front();
//    real& back();

//    // Modifiers
//    void pushBack(const real& val);
//    void pushBack(real&& val);
//    real popBack();
//    void clear() noexcept;

//private:
//    unsigned m_size;
//    std::vector<real> m_vec;

//};

#endif // VECTOR_H

































































