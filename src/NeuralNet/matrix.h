#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <vector>

class Vector;


class Matrix
{
public:
    Matrix();
    Matrix(unsigned rows, unsigned cols);
    Matrix(unsigned rows, unsigned cols, const real& initVal);
    Matrix(unsigned rows, unsigned cols, const Vector& initVals);
    Matrix(unsigned rows, unsigned cols, const std::vector<real>& initVals);
    Matrix(unsigned rows, unsigned cols, const std::initializer_list<real>& list);
    Matrix(const std::vector<real>& vec);
    Matrix(const std::initializer_list<real>& list);
    Matrix(const std::initializer_list<std::initializer_list<real>>& row_list);

    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs) const;                                         // vec: exactly same
    bool operator!=(const Matrix& rhs) const;                                         // vec: exactly same
    Matrix operator+(const Matrix& rhs) const;                                         // vec: exactly same
    Matrix operator-(const Matrix& rhs) const;                                        // vec: exactly same
    Matrix operator*(const Matrix& rhs) const;                                        // vec: exactly same
    Matrix& operator+=(const Matrix& rhs);                                      // vec: exactly same
    Matrix& operator-=(const Matrix& rhs);                                      // vec: exactly same
    Matrix& operator*=(const Matrix& rhs);                                      // vec: exactly same

    // Matrix/scalar operators
    Matrix operator+(const real& rhs) const;                                          // vec: exactly same
    Matrix operator-(const real& rhs) const;                                          // vec: exactly same
    Matrix operator*(const real& rhs) const;                                          // vec: exactly same
    Matrix operator/(const real& rhs) const;                                          // vec: exactly same
    Matrix& operator+=(const real& rhs);                                        // vec: exactly same
    Matrix& operator-=(const real& rhs);                                        // vec: exactly same
    Matrix& operator*=(const real& rhs);                                        // vec: exactly same
    Matrix& operator/=(const real& rhs);                                        // vec: exactly same

    // Matrix operators
    Matrix transpose() const;                                                         // vec: exactly same
    Matrix& multiplyElemWise(const Matrix& rhs);                                // vec: exactly same

    // Access
    std::vector<real>& vec();               // Remove, only access .data()      // vec: exactly same
    const std::vector<real>& vec() const;   // Remove, only access .data()      // vec: exactly same
//    real* data();
//    const real* data() const;
    unsigned rows() const;                                                      // vec: exactly same
    unsigned cols() const;                                                      // vec: exactly same
    unsigned size() const;                                                      // vec: exactly same
    real& operator()(const unsigned& row, const unsigned& col);                 // vec: delete
    real& operator()(const unsigned& idx);                                      // vec: exactly same
    const real& operator()(const unsigned& row, const unsigned& col) const;     // vec: delete
    const real& operator()(const unsigned& idx) const;                          // vec: exactly same

    Vector row(const unsigned& row) const;
    Vector col(const unsigned& row) const;

    // Modify
    // Need difference between add new row and edit exisiting row
    void addRow(const Vector& row);                                             // vec: delete
    void addCol(const Vector& col);                                             // vec: delete        

    // Utility
    std::string num2str(real num);                                              // vec: exactly same
    std::string str();                                                          // vec: exactly same

protected:
    unsigned m_rows;
    unsigned m_cols;
    std::vector<real> m_vec;

};

#endif // MATRIX_H


































































