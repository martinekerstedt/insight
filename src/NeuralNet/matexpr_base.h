#ifndef MATEXPR_BASE_H
#define MATEXPR_BASE_H

#include <Common/types.h>
#include <cstring>


// Forward declaration for Matrix
// so MatExpr can use it
class Matrix;
class Vector;
class VectorView;


// Forward declaration for Matrix Expressions
// so that Matrix can use them
class MatExpr {
//    virtual real operator()(const unsigned row, const unsigned col) const = 0;
//    virtual unsigned rows() const = 0;
//    virtual unsigned cols() const = 0;
//    virtual unsigned evalCost() const = 0;
//    virtual unsigned size() const = 0;
//    virtual bool sourceOk() = 0;
//    virtual void cache() = 0;
}; // Might add override requirements here


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


//template <class T>
//concept is_mat = std::is_base_of_v<Matrix, std::remove_cvref_t<T>>;

template <class T>
concept is_mat = std::is_base_of_v<Matrix, std::remove_cvref_t<T>>
                 || std::is_base_of_v<VectorView, std::remove_cvref_t<T>>;

template <class T>
concept is_expr =
        std::is_base_of_v<MatExpr, std::remove_cvref_t<T>>
        || is_mat<T>;



#endif // MATEXPR_BASE_H
