#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <Common/parallelfor.h>
#include <Matrix/expr_base.h>
#include <vector>


class Vector;
class VectorView;


class Matrix : public ExprInterface
{
public:
    // Constructors
    Matrix();
    Matrix(unsigned rows, unsigned cols);
    Matrix(unsigned rows, unsigned cols, const real& initVal);
    Matrix(unsigned rows, unsigned cols, const Vector& initVals);
    Matrix(unsigned rows, unsigned cols, const std::vector<real>& initVals);
    Matrix(unsigned rows, unsigned cols, const std::initializer_list<real>& list);
    Matrix(const std::vector<real>& vec);
    Matrix(const std::initializer_list<real>& list);
    Matrix(const std::initializer_list<std::initializer_list<real>>& row_list);

    template <Expr E>
    Matrix(const E& expr) :
        m_rows(expr.rows()),
        m_cols(expr.cols())
    {
        m_vec.resize(m_rows*m_cols);
        evalExpr(expr);
    }


    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs) const;
    bool operator!=(const Matrix& rhs) const;

    template <Expr RHS>
    Matrix& operator+=(const RHS& rhs)
    {
        *this = ExprAdd<Matrix, RHS>(*this, rhs);
        return (*this);
    }

    template <Expr RHS>
    Matrix& operator-=(const RHS& rhs)
    {
        *this = ExprSub<Matrix, RHS>(*this, rhs);
        return (*this);
    }

    template <Expr RHS>
    Matrix& operator*=(const RHS& rhs)
    {
        *this = ExprMatMul<Matrix, RHS>(*this, rhs);
        return (*this);
    }


    // Matrix operators
    ExprTrans<Matrix> trans() const;

    template <Expr LHS, Expr RHS>
    static auto mulEWise(const LHS& lhs, const RHS& rhs)
    {
        return ExprEWiseMul<LHS, RHS>(lhs, rhs);
    }


    // Matrix/scalar operators
    Matrix& operator+=(const real& rhs);
    Matrix& operator-=(const real& rhs);
    Matrix& operator*=(const real& rhs);
    Matrix& operator/=(const real& rhs);


    // Access
    std::vector<real>& vec();
    const std::vector<real>& vec() const;
    unsigned rows() const;
    unsigned cols() const;
    unsigned size() const;
    real& operator()(const unsigned& row, const unsigned& col);
    real& operator()(const unsigned& idx);
    real operator()(const unsigned row, const unsigned col) const;
    real operator()(const unsigned& idx) const;
    VectorView row(const unsigned& row) const;
//    Vector row(const unsigned& row) const;
//    Vector col(const unsigned& row) const;


    // Modify
    // Need difference between add new row and edit exisiting row
    void fill(real val);
    void reserve(unsigned size);
    void resize(unsigned rows, unsigned cols, real val = 0.0);
    void addRow(const Vector& row);
    void addCol(const Vector& col);


    // Utility
    std::string num2str(real num);
    std::string str();


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

    template <Expr RHS>
    Matrix operator=(const RHS& rhs)
    {
        if (m_rows != rhs.rows()) {
            THROW_ERROR("Matricies must have equal number of rows.\n"
                        << m_rows
                        << " != "
                        << rhs.rows());
        }

        if (m_cols != rhs.cols()) {
            THROW_ERROR("Matricies must have equal number of rows.\n"
                        << m_cols
                        << " != "
                        << rhs.cols());
        }

//        m_rows = rhs.rows();
//        m_cols = rhs.cols();
//        m_vec.resize(m_rows*m_cols);

        evalExpr(rhs);

        return *this;
    }

    template <Expr RHS>
    void evalExpr(const RHS& rhs)
    {
        // Cache if needed
        const_cast<RHS&>(rhs).sourceOk(*this);

        // Eval
        parallel_for([&](unsigned start, unsigned end)
        {
            for (unsigned i = start; i < end; ++i) {
                m_vec[i] = rhs(i / m_cols, i % m_cols);
            }
        }, m_rows*m_cols);
    }


    template<Expr E, class func>
    static auto apply(const E& expr, func f)
    {
        return ExprApply<E, func>(expr, f);
    }

    template<Expr E, class func, class... args>
    static auto apply(const E& expr, func f, const args&... a)
    {
        return ExprApply<E, func, args...>(expr, f, a...);
    }

    template<Expr LHS, Expr RHS, class func>
    static auto zip(const LHS& lhs, const RHS& rhs, func f)
    {
        return ExprZip<func, LHS, RHS>(f, rhs, lhs);
    }

    template<Expr LHS, Expr RHS, class func, class... args>
    static auto zip(const LHS& lhs, const RHS& rhs, func f, const args&... a)
    {
        return ExprZip<func, LHS, RHS, args...>(f, lhs, rhs, a...);
    }


protected:
    unsigned m_rows;
    unsigned m_cols;
    std::vector<real> m_vec;

};


// Include the implementation of the Matrix Expressions
// All template expression will now be in the same unit
// And only matrix.h has to be included by the client
#include <Matrix/expr.h>


#endif // MATRIX_H


































































