#ifndef VECTOR_H
#define VECTOR_H

#include <Common/types.h>
#include <vector>

class Matrix;

class Vector
{
public:
    Vector();
    Vector(unsigned size);
    Vector(unsigned size, const real& initval);
    Vector(const Vector& rhs);
    Vector(const Matrix& rhs);
    Vector(const std::vector<real>& rhs);    
    virtual ~Vector();

    // Assignment operators
    Vector& operator=(const Vector& rhs);
    Vector& operator=(const Matrix& rhs);
    Vector& operator=(const std::vector<real>& rhs);

    // Vector/Vector operators
    bool operator==(const Vector& rhs);
    bool operator!=(const Vector& rhs);
    Vector operator+(const Vector& rhs);
    Vector operator-(const Vector& rhs);
    Vector operator*(const Vector& rhs);
    Vector operator/(const Vector& rhs);
    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector& operator*=(const Vector& rhs);
    Vector& operator/=(const Vector& rhs);

    // Vector/std::vector operators
    bool operator==(const std::vector<real>& rhs);
    bool operator!=(const std::vector<real>& rhs);
    Vector operator+(const std::vector<real>& rhs);
    Vector operator-(const std::vector<real>& rhs);
    Vector operator*(const std::vector<real>& rhs);
    Vector operator/(const std::vector<real>& rhs);
    Vector& operator+=(const std::vector<real>& rhs);
    Vector& operator-=(const std::vector<real>& rhs);
    Vector& operator*=(const std::vector<real>& rhs);
    Vector& operator/=(const std::vector<real>& rhs);

    // Vector/scalar operators
    Vector operator+(const real& rhs);
    Vector operator-(const real& rhs);
    Vector operator*(const real& rhs);
    Vector operator/(const real& rhs);
    Vector& operator+=(const real& rhs);
    Vector& operator-=(const real& rhs);
    Vector& operator*=(const real& rhs);
    Vector& operator/=(const real& rhs);

    // Vector/Matrix operators
    bool operator==(const Matrix& rhs);
    bool operator!=(const Matrix& rhs);
    Vector operator*(const Matrix& rhs);
    Vector& operator*=(const Matrix& rhs);

    // Access
    std::vector<real>& vec();
    const std::vector<real>& vec() const;
    real& operator()(const unsigned& idx);
    const real& operator()(const unsigned& idx) const;
    unsigned size() const;
    real& front();
    real& back();

    // Modifiers
    void pushBack(const real& val);
    void pushBack(real&& val);
    real popBack();
    void clear() noexcept;

private:
    unsigned m_size;
    std::vector<real> m_vec;

};

#endif // VECTOR_H

































































