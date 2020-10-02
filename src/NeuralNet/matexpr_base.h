#ifndef MATEXPR_BASE_H
#define MATEXPR_BASE_H

#include <Common/types.h>
#include <cstring>


// Forward declaration for Matrix
// so MatExpr can use it
class Matrix;


// Forward declaration for Matrix Expressions
// so that Matrix can use them
struct MatExpr {}; // Might add override requirements here

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

// Is true for T = Matrix and T = Vector
template <class T>
constexpr bool is_mat = std::is_base_of_v<Matrix, std::remove_cvref_t<T> >;

template <class T>
constexpr bool is_expr =
        std::is_base_of_v<MatExpr, std::remove_cvref_t<T> >
        || is_mat<T>;



#endif // MATEXPR_BASE_H
