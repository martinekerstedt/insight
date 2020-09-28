#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <vector>
#include <NeuralNet/matexpr_base.h>

class Vector;

// TODO: Move sematics https://stackoverflow.com/questions/3106110/what-is-move-semantics

class Matrix : public MatExpr<Matrix>
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
    template <typename E>
    Matrix(const MatExpr<E>& expr) :
        m_rows(expr.rows()),
        m_cols(expr.cols())
    {
        m_vec.resize(m_rows*m_cols);

        const_cast<MatExpr<E>&>(expr).sourceOk(*this);

        for (unsigned row = 0; row < m_rows; ++row) {
            for (unsigned col = 0; col < m_cols; ++col) {
                m_vec[row*m_cols + col] = expr(row, col);
            }
        }
    }

    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs) const;                                         // vec: exactly same
    bool operator!=(const Matrix& rhs) const;                                         // vec: exactly same
    template <typename E>
    Matrix& operator+=(const MatExpr<E>& rhs)
    {
        // Might do 2 temp every time
        // Atleast one temp, since dest is always in expr
        *this = MatExprAdd<Matrix, E>(*this, *static_cast<const E*>(&rhs));
        return (*this);
    }

    template <typename E>
    Matrix& operator-=(const MatExpr<E>& rhs)
    {
        *this = MatExprSub<Matrix, E>(*this, *static_cast<const E*>(&rhs));
        return (*this);
    }

    template <typename E>
    Matrix& operator*=(const MatExpr<E>& rhs)
    {
        *this = MatExprMatMul<Matrix, E>(*this, *static_cast<const E*>(&rhs));
        return (*this);
    }

//    Matrix operator+(const Matrix& rhs) const;                                         // vec: exactly same
//    Matrix operator-(const Matrix& rhs) const;                                        // vec: exactly same
//    Matrix operator*(const Matrix& rhs) const;                                        // vec: exactly same
//    Matrix& operator+=(const Matrix& rhs);                                      // vec: exactly same
//    Matrix& operator-=(const Matrix& rhs);                                      // vec: exactly same
//    Matrix& operator*=(const Matrix& rhs);                                      // vec: exactly same

    // Matrix/scalar operators
//    Matrix operator+(const real& rhs) const;                                          // vec: exactly same
//    Matrix operator-(const real& rhs) const;                                          // vec: exactly same
//    Matrix operator*(const real& rhs) const;                                          // vec: exactly same
//    Matrix operator/(const real& rhs) const;                                          // vec: exactly same
    Matrix& operator+=(const real& rhs);                                        // vec: exactly same
    Matrix& operator-=(const real& rhs);                                        // vec: exactly same
    Matrix& operator*=(const real& rhs);                                        // vec: exactly same
    Matrix& operator/=(const real& rhs);                                        // vec: exactly same

    // Matrix operators
    MatExprTrans<Matrix> trans() const;                                                       // vec: exactly same

    template <typename E>
    auto mulEWise(const MatExpr<E>& rhs)
    {
        return MatExprEWiseMul<Matrix, E>(*this, *static_cast<const E*>(&rhs));
    }

//    Matrix transpose() const;                                                         // vec: exactly same
//    Matrix& multiplyElemWise(const Matrix& rhs);                                // vec: exactly same

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
    void reserve(unsigned size);
    void resize(unsigned rows, unsigned cols);
    void resize(unsigned size);
    void addRow(const Vector& row);                                             // vec: delete
    void addCol(const Vector& col);                                             // vec: delete        

    // Utility
    std::string num2str(real num);                                              // vec: exactly same
    std::string str();                                                          // vec: exactly same

    // Lazy evaluation assignment
    bool sourceOk(const Matrix& destMat);

    constexpr unsigned evalCost() const
    {
        return 1;
    }

    template <typename E>
    Matrix operator=(const MatExpr<E>& rhs)
    {
        m_rows = rhs.rows();
        m_cols = rhs.cols();
        m_vec.resize(m_rows*m_cols);

        const_cast<MatExpr<E>&>(rhs).sourceOk(*this);

        for (unsigned row = 0; row < m_rows; ++row) {
            for (unsigned col = 0; col < m_cols; ++col) {
                m_vec[row*m_cols + col] = rhs(row, col);
            }
        }

        return *this;
    }

protected:
    unsigned m_rows;
    unsigned m_cols;
    std::vector<real> m_vec;

};


// Include the implementation of the Matrix Expressions
// All template expression will now be in the same unit
// And only matrix.h has to be included by the client
#include <NeuralNet/matexpr.h>


#endif // MATRIX_H


































































