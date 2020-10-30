#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <Common/parallelfor.h>
#include <vector>
#include <NeuralNet/matexpr_base.h>


#include <thread>
#include <algorithm>

class Vector;

// TODO: Move sematics https://stackoverflow.com/questions/3106110/what-is-move-semantics

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

    template <class E> requires(is_expr<E>)
    Matrix(const E& expr) :
        m_rows(expr.rows()),
        m_cols(expr.cols())
    {
        m_vec.resize(m_rows*m_cols);
        evalExpr(expr);
    }

    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs) const;                                         // vec: exactly same
    bool operator!=(const Matrix& rhs) const;                                         // vec: exactly same

    template <class E> requires(is_expr<E>)
    Matrix& operator+=(const E& rhs)
    {
        *this = MatExprAdd<Matrix, E>(*this, rhs);
        return (*this);
    }

    template <class E> requires(is_expr<E>)
    Matrix& operator-=(const E& rhs)
    {
        *this = MatExprSub<Matrix, E>(*this, rhs);
        return (*this);
    }

    template <class E> requires(is_expr<E>)
    Matrix& operator*=(const E& rhs)
    {
        *this = MatExprMatMul<Matrix, E>(*this, rhs);
        return (*this);
    }


    // Matrix/scalar operators
    Matrix& operator+=(const real& rhs);                                        // vec: exactly same
    Matrix& operator-=(const real& rhs);                                        // vec: exactly same
    Matrix& operator*=(const real& rhs);                                        // vec: exactly same
    Matrix& operator/=(const real& rhs);                                        // vec: exactly same

    // Matrix operators
    MatExprTrans<Matrix> trans() const;                                                       // vec: exactly same

    template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
    static auto mulEWise(const E1& lhs, const E2& rhs)
    {
        return MatExprEWiseMul<E1, E2>(lhs, rhs);
    }


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

    // Lazy evaluation
    bool sourceOk(const Matrix& destMat);

    constexpr void cache()
    {
        return;
    }

    constexpr unsigned evalCost() const
    {
        return 1;
    }

    template <class E> requires(is_expr<E>)
    Matrix operator=(const E& rhs)
    {
        m_rows = rhs.rows();
        m_cols = rhs.cols();
        m_vec.resize(m_rows*m_cols);

        evalExpr(rhs);

        return *this;
    }

    template <class E> requires(is_expr<E>)
    void evalExpr(const E& rhs)
    {
        // Cache if needed
        const_cast<E&>(rhs).sourceOk(*this);

        // Eval
        parallel_for([&](unsigned start, unsigned end)
        {
            for (unsigned i = start; i < end; ++i) {
                m_vec[i] = rhs(i / m_cols, i % m_cols);
            }
        }, m_rows*m_cols);
    }


    template<class E, class func> requires(is_expr<E>)
    static auto apply(const E& expr, func f)
    {
        return MatExprApply<E, func>(expr, f);
    }

    template<class E, class func, class... args> requires(is_expr<E>)
    static auto apply(const E& expr, func f, const args&... a)
    {
        return MatExprApply<E, func, args...>(expr, f, a...);
    }

    template<class E1, class E2, class func> requires(is_expr<E1> && is_expr<E2>)
    static auto zip(const E1& lhs, const E2& rhs, func f)
    {
        return MatExprZip<E1, E2, func>(lhs, rhs, f);
    }

    template<class E1, class E2, class func, class... args> requires(is_expr<E1> && is_expr<E2>)
    static auto zip(const E1& lhs, const E2& rhs, func f, const args&... a)
    {
        return MatExprZip<E1, E2, func, args...>(lhs, rhs, f, a...);
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


































































