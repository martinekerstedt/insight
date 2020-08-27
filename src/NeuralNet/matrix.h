#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <NeuralNet/vector.h>
#include <vector>

class Matrix
{
public:
    Matrix();
    Matrix(unsigned rows, unsigned cols);
    Matrix(unsigned rows, unsigned cols, const real& initVal);
    Matrix(unsigned rows, unsigned cols, const Vector& initVals);
    Matrix(unsigned rows, unsigned cols, const std::vector<real>& initVals);
    Matrix(const Matrix& mat);
    Matrix(const Vector& vec);    
    Matrix(const std::vector<real>& vec);

    virtual ~Matrix();

    // Assignment operators
    Matrix& operator=(const Matrix& rhs);
    Matrix& operator=(const Vector& rhs);
    Matrix& operator=(const std::vector<real>& rhs);

    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs);
    bool operator!=(const Matrix& rhs);
    Matrix operator+(const Matrix& rhs);
    Matrix operator-(const Matrix& rhs);
    Matrix operator*(const Matrix& rhs);
    Matrix& operator+=(const Matrix& rhs);    
    Matrix& operator-=(const Matrix& rhs);    
    Matrix& operator*=(const Matrix& rhs);

    // Matrix/Vector operators
    bool operator==(const Vector& rhs);
    bool operator!=(const Vector& rhs);
    Vector operator*(const Vector& rhs);
    Matrix addRowWise(const Vector& rhs);
    Matrix addColWise(const Vector& rhs);
    Matrix subtractRowWise(const Vector& rhs);
    Matrix subtractColWise(const Vector& rhs);
    Matrix multiplyRowWise(const Vector& rhs);
    Matrix multiplyColWise(const Vector& rhs);
    Matrix divideRowWise(const Vector& rhs);
    Matrix divideColWise(const Vector& rhs);

    // Matrix/std::vector operators
    bool operator==(const std::vector<real>& rhs);
    bool operator!=(const std::vector<real>& rhs);
    Vector operator*(const std::vector<real>& rhs);

    // Matrix/scalar operators
    Matrix operator+(const real& rhs);
    Matrix operator-(const real& rhs);
    Matrix operator*(const real& rhs);
    Matrix operator/(const real& rhs);
    Matrix& operator+=(const real& rhs);
    Matrix& operator-=(const real& rhs);
    Matrix& operator*=(const real& rhs);
    Matrix& operator/=(const real& rhs);

    // Matrix operators
    Matrix transpose();
    Matrix subtractElemWise(const Matrix& rhs);
    Matrix multiplyElemWise(const Matrix& rhs);

    // Access
    std::vector<real>& vec();
    const std::vector<real>& vec() const;
    unsigned rows() const;
    unsigned cols() const;
    unsigned size() const;
    real& operator()(const unsigned& row, const unsigned& col);
    real& operator()(const unsigned& idx);
    const real& operator()(const unsigned& row, const unsigned& col) const;    
    const real& operator()(const unsigned& idx) const;

    // Print
    std::string num2str(real num);
    std::string str();

private:
    unsigned m_rows;
    unsigned m_cols;
    std::vector<real> m_vec;

};

#endif // MATRIX_H

































































