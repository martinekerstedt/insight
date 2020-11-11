#ifndef EXPR_BASE_H
#define EXPR_BASE_H

#include <Common/types.h>
#include <tuple>


// Forward declaration of Matrix so Expr can use it
class Matrix;


// Interface that all Expressions must support
struct ExprInterface
{
    virtual real operator()(const unsigned row, const unsigned col) const = 0;
    virtual unsigned rows() const = 0;
    virtual unsigned cols() const = 0;
    virtual unsigned evalCost() const = 0;
    virtual unsigned size() const = 0;
    virtual bool sourceOk(const Matrix& destMat) = 0;
    virtual void cache() = 0;
};


// Common structure that all non-container expressions have
struct ExprBase : private ExprInterface
{
    virtual real operator()(const unsigned row, const unsigned col) const = 0;
    virtual unsigned rows() const = 0;
    virtual unsigned cols() const = 0;
    virtual unsigned evalCost() const = 0;
    virtual bool sourceOk(const Matrix& destMat) = 0;

    unsigned size() const
    {
        return rows()*cols();
    }

    void cache()
    {
        m_tmp.reserve(rows()*cols());

        for (unsigned row = 0; row < rows(); ++row) {
            for (unsigned col = 0; col < cols(); ++col) {
                m_tmp[row*cols() + col] = operator()(row, col);
            }
        }

        m_cached = true;
    }

    bool m_cached = false;
    std::vector<real> m_tmp;

};


// Proxy type for element wise multiplication, mat ** mat
template <class E>
struct ExprEWiseMulProxy;


// Expression concept, defines what an Expression is
template <class T>
concept Expr =
        (std::is_base_of_v<ExprInterface, std::remove_cvref_t<T>>
        || std::is_same_v<ExprEWiseMulProxy, std::remove_cvref_t<T>>)
        && !std::is_pointer_v<T>;

template <class T>
concept is_expr = Expr<T>;


// Forward declaration of Expression types so Matrix can use them
// Requires: (args[0] == Expr) || ((args[0] == arithmetic) && (args[1] == Expr))
template<class func, class... args>
requires((is_expr<typename std::tuple_element<0,std::tuple<args...>>::type>)
         || ((std::is_arithmetic_v<typename std::tuple_element<0,std::tuple<args...>>::type>)
             && (is_expr<typename std::tuple_element<1,std::tuple<args...>>::type>)))
struct ExprZip;

template <class LHS, class RHS>
struct ExprAdd;

template <class LHS, class RHS>
struct ExprSub;

template <class LHS, class RHS>
struct ExprDiv;

template <class LHS, class RHS>
struct ExprEWiseMul;

template<class func, class E, class... args>
struct ExprApply;

template <class LHS, class RHS>
struct ExprMatMul;

template <class E>
struct ExprTrans;




#endif // EXPR_BASE_H


















