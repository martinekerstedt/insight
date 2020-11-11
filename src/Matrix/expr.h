#ifndef EXPR_H
#define EXPR_H

#include <Matrix/expr_base.h>
#include <cassert>


// Zip
//
// All unary and binary matrix operations
// where the operand matrix/matrices and the result matrix have the same size
//
// Requires: (args[0] == Expr) || ((args[0] == real) && (args[1] == Expr))
template<class func, class... args>
requires((is_expr<typename std::tuple_element<0,std::tuple<args...>>::type>)
         || ((std::is_arithmetic_v<typename std::tuple_element<0,std::tuple<args...>>::type>)
             && (is_expr<typename std::tuple_element<1,std::tuple<args...>>::type>)))
struct ExprZip : private ExprBase
{
    using a0_t = typename std::tuple_element<0,std::tuple<args...>>::type;
    using a1_t = typename std::tuple_element<1,std::tuple<args...>>::type;

    ExprZip(func f, const args&... a) :
        m_func(f),
        m_args(const_cast<args&>(a)...) {}

    real operator()(const unsigned row, const unsigned col) const
    {
        if (this->m_cached) {

            return this->m_tmp[row*cols() + col];

        } else {

            if constexpr (sizeof...(args) == 1) {

                // One arg, that is an Expr
                return m_func((std::get<0>(m_args))(row, col));

            } else if constexpr (sizeof...(args) == 2) {

                if constexpr (is_expr<a0_t>) {

                    if constexpr (is_expr<a1_t>) {

                        // Two args, that both are Expr
                        return m_func((std::get<0>(m_args))(row, col),
                                      (std::get<1>(m_args))(row, col));

                    } else {

                       // Two args, only the first is Expr
                       return m_func((std::get<0>(m_args))(row, col),
                                     std::get<1>(m_args));
                    }

                } else {

                    // Two args, first is arithmetic, second must be Expr
                    return m_func(std::get<0>(m_args),
                                  (std::get<1>(m_args))(row, col));
                }

            } else {

                if constexpr (is_expr<a0_t>) {

                    if constexpr (is_expr<a1_t>) {

                        // First two args are Expr
                        return apply2((std::get<0>(m_args)(row, col)),
                                      (std::get<1>(m_args)(row, col)),
                                      std::make_index_sequence<sizeof...(args) - 2>{});

                    } else {

                       // First arg is Expr
                        return apply1((std::get<0>(m_args)(row, col)),
                                      std::make_index_sequence<sizeof...(args) - 1>{});
                    }

                } else {

                    // First is arithmetic, second must be Expr
                    return apply2(std::get<0>(m_args),
                                  (std::get<1>(m_args)(row, col)),
                                  std::make_index_sequence<sizeof...(args) - 2>{});
                }
            }
        }
    }

    template<std::size_t... Is>
    auto apply1(real a, std::index_sequence<Is...>) const
    {
        return m_func(a, std::get<1 + Is>(m_args)...);
    }

    template<std::size_t... Is>
    auto apply2(real a, real b, std::index_sequence<Is...>) const
    {
        return m_func(a, b, std::get<2 + Is>(m_args)...);
    }

    unsigned rows() const
    {
        if constexpr (is_expr<a0_t>) {
            // First arg is Expr
            return (std::get<0>(m_args)).rows();
        } else {
            return (std::get<1>(m_args)).rows();
        }
    }

    unsigned cols() const
    {
        if constexpr (is_expr<a0_t>) {
            // First arg is Expr
            return (std::get<0>(m_args)).cols();
        } else {
            return (std::get<1>(m_args)).cols();
        }
    }

    unsigned evalCost() const
    {
        // No additional cost
        return 1;
    }

    bool sourceOk(const Matrix& destMat)
    {
        if constexpr (is_expr<a0_t>) {

            if constexpr (is_expr<a1_t>) {
                // Both is Expr
                return (std::get<0>(m_args)).sourceOk(destMat)
                        || (std::get<1>(m_args)).sourceOk(destMat);
            } else {
                // Only first arg is Expr
                return (std::get<0>(m_args)).sourceOk(destMat);
            }

        } else {

            // Second arg is Expr
            return (std::get<1>(m_args)).sourceOk(destMat);
        }
    }

private:
    func m_func;
    std::tuple<args&...> m_args;

};


// Addition
template <class LHS, class RHS>
struct ExprAdd final : public ExprZip<real (*)(const real&, const real&), LHS, RHS>
{
    ExprAdd(const LHS& lhs, const RHS& rhs) :
        ExprZip<real (*)(const real&, const real&), LHS, RHS>
        ([](const real& a, const real& b){return a+b;}, lhs, rhs) {}
};


// Subtraction
template <class LHS, class RHS>
struct ExprSub final : public ExprZip<real (*)(const real&, const real&), LHS, RHS>
{
    ExprSub(const LHS& lhs, const RHS& rhs) :
        ExprZip<real (*)(const real&, const real&), LHS, RHS>
        ([](const real& a, const real& b){return a-b;}, lhs, rhs) {}
};


// Division
template <class LHS, class RHS>
struct ExprDiv final : public ExprZip<real (*)(const real&, const real&), LHS, RHS>
{
    ExprDiv(const LHS& lhs, const RHS& rhs) :
        ExprZip<real (*)(const real&, const real&), LHS, RHS>
        ([](const real& a, const real& b){return a/b;}, lhs, rhs) {}
};


// Element wise multiplication
template <class LHS, class RHS>
struct ExprEWiseMul final : public ExprZip<real (*)(const real&, const real&), LHS, RHS>
{
    ExprEWiseMul(const LHS& lhs, const RHS& rhs) :
        ExprZip<real (*)(const real&, const real&), LHS, RHS>
        ([](const real& a, const real& b){return a*b;}, lhs, rhs) {}
};

template <class E>
struct ExprEWiseMulProxy
{
    const E& expr;   
};


// Apply
template<class E, class func, class... args>
struct ExprApply final : public ExprZip<func, E, args...>
{
    ExprApply(const E& lhs, func f, const args&... a) :
        ExprZip<func, E, args...>(f, lhs, a...) {}
};


// Martrix multiplication
template <class LHS, class RHS>
struct ExprMatMul final : private ExprBase
{
    ExprMatMul(const LHS& lhs, const RHS& rhs) :
        m_lhs(lhs),
        m_rhs(rhs)
    {
        assert(lhs.cols() == rhs.rows());
    }

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp[row*cols() + col];
        } else {
            float sum = 0;

            // lhs.cols() == rhs.rows()
            for (unsigned j = 0; j < m_lhs.cols(); ++j) {
                sum +=  m_lhs(row, j) * m_rhs(j, col);
            }

            return sum;
        }
    }

    unsigned rows() const
    {
        return m_lhs.rows();
    }

    unsigned cols() const
    {
        return m_rhs.cols();
    }

    bool sourceOk(const Matrix& destMat)
    {
        if (!const_cast<LHS&>(m_lhs).sourceOk(destMat)
                || !const_cast<RHS&>(m_rhs).sourceOk(destMat)) {
            cache();
        }

        return true;
    }

    unsigned evalCost() const
    {        
        return 1;
    }

private:
    const LHS& m_lhs;
    const RHS& m_rhs;

};


// Unary expressions
// Transposition
template <class E>
struct ExprTrans final : private ExprBase
{
    // For very large matrices,
    // it is more efficient to actually do the transpose,
    // because of how spread the memory access is otherwise.
    // Create a temp and wirte the transpose to it
    // Need to overload cache() to make it optimized

    ExprTrans(const E& expr) :
        m_expr(expr) {}

    real operator()(const unsigned row, const unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp[row*cols() + col];
        } else {
            return m_expr(col, row);
        }
    }

    unsigned rows() const
    {
        return m_expr.cols();
    }

    unsigned cols() const
    {
        return m_expr.rows();
    }

    unsigned evalCost() const
    {
        return 1;
    }

    bool sourceOk(const Matrix& destMat)
    {
        if (!const_cast<E&>(m_expr).sourceOk(destMat)) {
            cache();
        }

        return true;
    }

private:
    const E& m_expr;

};


// Is evalCost the cost of evaluating the whole expression or just one element?
//
//
// evalCost is the cost of evaluation the whole expression
//
// Cached operands
//
// return size()*m_lhs.cols()                           // Number of operations for current op
//         * 1                                          // Cost of one m_lhs(row, col)
//         * 1                                          // Cost of one m_lhs(row, col)
//         + m_lhs.evalCost()                           // Cost of caching operands
//         + m_rhs.evalCost();                          // Cost of caching operands
//
//
// Non cached operands
//
// return size()*m_lhs.cols()                           // Number of operations for current op
//         * (m_lhs.evalCost()/m_lhs.size())            // Cost of one m_lhs(row, col)
//         * (m_rhs.evalCost()/m_rhs.size())            // Cost of one m_rhs(row, col)
//
//
//
// evalCost is the cost of evaluation just one element
//
// Cached operands
//
// return m_lhs.cols()                                  // Number of operations for evaluating one elem with current op
//         * 1                                          // Cost of one m_lhs(row, col)
//         * 1                                          // Cost of one m_lhs(row, col)
//         + (m_lhs.evalCost()*m_lhs.size()) / size()   // Cost of caching operands
//         + (m_rhs.evalCost()*m_rhs.size()) / size()   // Cost of caching operands
//
// Per access to current op; cost of current op, number of ops per, (m_lhs.evalCost()*m_lhs.size()) / size()
//
//
// Non cached operands
//
// return m_lhs.cols()                                  // Number of operations for evaluating one elem with current op
//         * m_lhs.evalCost()                           // Cost of one m_lhs(row, col)
//         * m_rhs.evalCost()                           // Cost of one m_lhs(row, col)
//
// Per access to current op; cost of access to operands, cost of current op, number of ops per
//
//  this.cost1 = lhs.cost and rhs.cost
//  this.cost2 = lhs.cachedCost and rhs.cost
//  this.cost3 = lhs.cost and rhs.cachedCost
//  this.cost4 = lhs.cachedCost and rhs.cachedCost
//
// Need all four values, then cache accordingly
//

//unsigned evalCost() const
//{
    // INCLUDE COST OF DOING THE ACTUAL CACHING
    // Cost of malloc of the tmp, cost of copy, cost of eval

//        if constexpr (std::tuple_size_v<std::tuple<const operands&...> > == 1) {
//            // Unary

//        } else {
//            // Binary

//            return m_costFunc(std::get<0>(m_args).evalCost(), std::get<1>(m_args).evalCost());
//        }

//    return 1;
//}




// Operators
// Addition
template <Expr LHS, Expr RHS>
auto operator+(const LHS& lhs, const RHS& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return ExprAdd<LHS, RHS>(lhs, rhs);
}

template <Expr LHS>
auto operator+(const LHS& lhs, const real& rhs)
{
    return ExprAdd<LHS, real>(lhs, rhs);
}

template <Expr RHS>
auto operator+(const real& lhs, const RHS& rhs)
{
    return ExprAdd<real, RHS>(lhs, rhs);
}


// Subtraction
template <Expr LHS, Expr RHS>
auto operator-(const LHS& lhs, const RHS& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return ExprSub<LHS, RHS>(lhs, rhs);
}

template <Expr LHS>
auto operator-(const LHS& lhs, const real& rhs)
{
    return ExprSub<LHS, real>(lhs, rhs);
}

template <Expr RHS>
auto operator-(const real& lhs, const RHS& rhs)
{
    return ExprSub<real, RHS>(lhs, rhs);
}


// Division
template <Expr LHS, Expr RHS>
auto operator/(const LHS& lhs, const RHS& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return ExprDiv<LHS, RHS>(lhs, rhs);
}

template <Expr LHS>
auto operator/(const LHS& lhs, const real& rhs)
{
    return ExprDiv<LHS, real>(lhs, rhs);
}

template <Expr RHS>
auto operator/(const real& lhs, const RHS& rhs)
{
    return ExprDiv<real, RHS>(lhs, rhs);
}


// Matrix element wise multiplication, mat ** mat
template <Expr LHS, Expr RHS>
auto operator*(const LHS& lhs, const ExprEWiseMulProxy<RHS>& rhs)
{
    assert(lhs.rows() == rhs.expr.rows());
    assert(lhs.cols() == rhs.expr.cols());
    return ExprEWiseMul<LHS, RHS>(lhs, rhs.expr);
}

template <Expr RHS>
auto operator*(const RHS& rhs)
{
    return ExprEWiseMulProxy<RHS>(rhs);
}

template <Expr LHS>
auto operator*(const LHS& lhs, const real& rhs)
{
    return ExprEWiseMul<LHS, real>(lhs, rhs);
}

template <Expr RHS>
auto operator*(const real& lhs, const RHS& rhs)
{
    return ExprEWiseMul<real, RHS>(lhs, rhs);
}


// Matrix multiplication
template <Expr LHS, Expr RHS>
auto operator*(const LHS& lhs, const RHS& rhs)
{
    assert(lhs.cols() == rhs.rows());
    return ExprMatMul<LHS, RHS>(lhs, rhs);
}




#endif // EXPR_H





















