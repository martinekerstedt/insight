#ifndef MATEXPR_BASE_H
#define MATEXPR_BASE_H

#include <Common/types.h>
#include <cstring>


// Forward declaration for Matrix
// so MatExpr can use it
class Matrix;



struct RequireOverride {
    virtual real operator()(const unsigned row, const unsigned col) const = 0;
    virtual unsigned rows() const = 0;
    virtual unsigned cols() const = 0;
    virtual bool sourceOk(const Matrix& destMat) = 0;
    virtual unsigned evalCost() const = 0;
};


template <typename E>
class MatExpr
{

public:
    real operator()(const unsigned row, const unsigned col) const
    {
        return static_cast<const E&>(*this)(row, col);
    }

    unsigned size() const
    {
        return static_cast<const E&>(*this).rows()*static_cast<const E&>(*this).cols();
    }

    unsigned rows() const
    {
        return static_cast<const E&>(*this).rows();
    }

    unsigned cols() const
    {
        return static_cast<const E&>(*this).cols();
    }

    bool sourceOk(const Matrix& destMat)
    {
        return static_cast<E&>(*this).sourceOk(destMat);
    }

    unsigned evalCost() const
    {
        return static_cast<const E&>(*this).evalCost();
    }
};


// Forward declaration for Matrix Expressions
// so that Matrix can use them
template <class E1, class E2>
class MatExprAdd;

template <class E1, class E2>
struct MatExprSub;

template <class E1, class E2>
struct MatExprEWiseMul;

template <class E1, class E2>
class MatExprMatMul;

template <class E>
class MatExprTrans;

template<class func, class E, class... args>
class MatExprApply;

template<class func, class E1, class E2, class... args>
class MatExprZip;


// Helper functions to check if a type is a MatExpr
//template<typename E>
//struct is_matexpr_impl : std::false_type {};

//template<typename E>
//struct is_matexpr_impl<MatExpr<E> > : std::true_type {};

//template<typename E>
//inline constexpr bool is_matexpr = is_matexpr_impl<E>::value;







#endif // MATEXPR_BASE_H
