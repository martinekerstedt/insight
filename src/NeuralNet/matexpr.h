﻿#ifndef MATEXPR_H
#define MATEXPR_H

#include <cassert>


// Only needed to make the code model work
// This include is not actually active
// since this file is included into mat.h
// where the header guard will stop this include
#include <NeuralNet/matrix.h>



// Need a map function, to deal with activation functions
// Should have an argument to state the complexity of the function
// Will default to high complexity

// NSTO = Non singel traversal operation

// Cache an operand to a NSTO if it involves computation that is above complexity
// threshold.
// A plain matrix doesn't. And neither does a transpose, if source ok.

// Cache the result of a NSTO if either operand contain the destination

template <bool sto, class E1, class E2 = void*>
class MatExprBase : public virtual MatExpr
{

public:
    MatExprBase(const E1& lhs, const E2& rhs) :
        m_lhs(lhs),
        m_rhs(rhs),
        m_cached(false) {}

    // Fix with std::optional
    MatExprBase(const E1& lhs) :
        m_lhs(lhs),
        m_rhs(nullptr),
        m_cached(false) {}

    virtual real operator()(const unsigned row, const unsigned col) const = 0;
    virtual unsigned rows() const = 0;
    virtual unsigned cols() const = 0;
    virtual unsigned evalCost() const = 0;

    unsigned size() const
    {
        return rows()*cols();
    }

    bool sourceOk(const Matrix& destMat)
    {
        if constexpr (std::is_same_v<E2, void*>) {

            // Unary expr

            if constexpr (std::is_arithmetic_v<E1>) {
                return true;
            } else {

                if constexpr (sto) {

                    return const_cast<E1&>(m_lhs).sourceOk(destMat);

                } else {

                    if (!const_cast<E1&>(m_lhs).sourceOk(destMat)) {
                        cache();
                    }

                    return true;
                }
            }

        } else {

            // Binary expr

            if constexpr (sto) {

                if constexpr (std::is_arithmetic_v<E1>) {
                    return const_cast<E2&>(m_rhs).sourceOk(destMat);
                } else if constexpr (std::is_arithmetic_v<E2>) {
                    return const_cast<E1&>(m_lhs).sourceOk(destMat);
                } else {
                    return const_cast<E1&>(m_lhs).sourceOk(destMat) || const_cast<E2&>(m_rhs).sourceOk(destMat);
                }

            } else {

                if constexpr (std::is_arithmetic_v<E1>) {
                    if (!const_cast<E2&>(m_rhs).sourceOk(destMat)) {
                        cache();
                    }

                    return true;

                } else if constexpr (std::is_arithmetic_v<E2>) {
                    if (!const_cast<E1&>(m_lhs).sourceOk(destMat)) {
                        cache();
                    }

                    return true;

                } else {
                    if (!const_cast<E1&>(m_lhs).sourceOk(destMat)
                            || !const_cast<E2&>(m_rhs).sourceOk(destMat)) {
                        cache();
                    }

                    return true;
                }
            }
        }
    }

    void cache()
    {
        m_tmp.resize(rows(), cols());

        for (unsigned row = 0; row < rows(); ++row) {
            for (unsigned col = 0; col < cols(); ++col) {
                m_tmp(row, col) = operator()(row, col);
            }
        }

        m_cached = true;
    }

protected:
    const E1& m_lhs;
    const E2& m_rhs;
    bool m_cached;
    Matrix m_tmp;

};

// Binary expressions
// Addition
template <class E1, class E2>
class MatExprAdd : public MatExprBase<true, E1, E2>, virtual MatExpr
{

public:
    MatExprAdd(const E1& lhs, const E2& rhs) :
        MatExprBase<true, E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            if constexpr (std::is_arithmetic_v<E1>) {
                return this->m_lhs + this->m_rhs(row, col);
            } else if constexpr (std::is_arithmetic_v<E2>) {
                return this->m_lhs(row, col) + this->m_rhs;
            } else {
                return this->m_lhs(row, col) + this->m_rhs(row, col);
            }
        }
    }

    unsigned rows() const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return this->m_rhs.rows();
        } else {
            return this->m_lhs.rows();
        }
    }

    unsigned cols() const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return this->m_rhs.cols();
        } else {
            return this->m_lhs.cols();
        }
    }

    unsigned evalCost() const
    {
        // Need to benchmark these to get values
        // Cost/st is probably reduced as size grows,
        // perhaps not for memory
        enum class Cost
        {
            ADD = 1,
            SUB = 1,
            MUL = 4,
            DIV = 4,
            ALLOC_REAL = 50,
            COPY_REAL = 50
        };

//        unsigned costOfOp = 1;
//        unsigned costAlloc = 1;
//        unsigned costAccessE1 = this->m_lhs.evalCost();
//        unsigned costAccessE2 = this->m_rhs.evalCost();
//        unsigned sizeE1 = this->m_lhs.size();
//        unsigned sizeE2 = this->m_rhs.size();
//        unsigned costCacheE1 = sizeE1 * (costAccessE1 * costAlloc);                                 // Total cost of caching E1, resulting in costAccessE1 = 1
//        unsigned costCacheE2 = sizeE2 * (costAccessE2 * costAlloc);                                 // Total cost of caching E2, resulting in costAccessE2 = 1

//        unsigned cacheE1cacheE2 = costOfOp + costCacheE1/this->size() + costCacheE2/this->size();
//        unsigned notCacheE1cacheE2 = costOfOp + costAccessE1 + costCacheE2/this->size();
//        unsigned cacheE1notCacheE2 = costOfOp + costCacheE1/this->size() + costAccessE2;
//        unsigned notCacheE1notCacheE2 = costOfOp + costAccessE1 + costAccessE2;

        return 1;
    }
};


// Subtraction
template <class E1, class E2>
struct MatExprSub : public MatExprAdd<E1, E2>, virtual MatExpr
{
    MatExprSub(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            if constexpr (std::is_arithmetic_v<E2>) {
                return this->m_lhs(row, col) - this->m_rhs;
            } else {
                return this->m_lhs(row, col) - this->m_rhs(row, col);
            }
        }
    }
};


// Division, only defined for Mat / Num
template <class E1, class E2>
struct MatExprDiv : public MatExprAdd<E1, E2>, virtual MatExpr
{
    MatExprDiv(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            return this->m_lhs(row, col) / this->m_rhs;
        }
    }
};


// Element wise multiplication
template <class E1, class E2>
struct MatExprEWiseMul : public MatExprAdd<E1, E2>, virtual MatExpr
{
    MatExprEWiseMul(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            if constexpr (std::is_arithmetic_v<E1>) {
                return this->m_lhs * this->m_rhs(row, col);
            } else if constexpr (std::is_arithmetic_v<E2>) {
                return this->m_lhs(row, col) * this->m_rhs;
            } else {
                return this->m_lhs(row, col) * this->m_rhs(row, col);
            }
        }
    }
};


// Martrix multiplication
template <class E1, class E2>
class MatExprMatMul : public MatExprBase<false, E1, E2>, virtual MatExpr
{

public:
    MatExprMatMul(const E1& lhs, const E2& rhs) :
        MatExprBase<false, E1, E2>(lhs, rhs)
    {
        static_assert (!std::is_arithmetic_v<E1> && !std::is_arithmetic_v<E2>, "Must be MatExpr");
        assert(lhs.cols() == rhs.rows());
    }

    real operator()(unsigned row, unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            float sum = 0;

            // lhs.cols() == rhs.rows()
            for (unsigned j = 0; j < this->m_lhs.cols(); ++j) {
                sum +=  this->m_lhs(row, j) * this->m_rhs(j, col);
            }

            return sum;
        }
    }

    unsigned rows() const
    {
        return this->m_lhs.rows();
    }

    unsigned cols() const
    {
        return this->m_rhs.cols();
    }

    unsigned evalCost() const
    {        
        return 1;
    }
};

template <class E>
class MatExprEWiseMul2 : public virtual MatExpr
{

public:
    MatExprEWiseMul2(const E& e) :
        expr(e) {}

    const E& expr;

};



// Unary expressions
// Transposition
template <class E>
class MatExprTrans : public MatExprBase<false, E>, virtual MatExpr
{

    // For very large matrices,
    // it is more efficient to actually do the transpose,
    // because of how spread the memory access is otherwise.
    // Create a temp and wirte the transpose to it
    // Need to overload cache() to make it optimized

public:
    MatExprTrans(const E& expr) :
        MatExprBase<false, E>(expr) {}

    real operator()(const unsigned row, const unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {
            return this->m_lhs(col, row);
        }
    }

    unsigned rows() const
    {
        return this->m_lhs.cols();
    }

    unsigned cols() const
    {
        return this->m_lhs.rows();
    }

    unsigned evalCost() const
    {
        return 1;
    }

};


// Apply
template<class E, class func, class... args>
class MatExprApply : public MatExprBase<true, E>, virtual MatExpr
{
//    using args_seq = std::index_sequence_for<args...>;

public:
    MatExprApply(const E& expr, func f, const args&... a) :
        MatExprBase<true, E>(expr),
        m_func(f),
        m_args(const_cast<args&>(a)...) {}

    real operator()(const unsigned row, const unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {

            if constexpr (sizeof... (args) == 0) {

                return m_func(this->m_lhs(row, col));
            } else {

                return apply(this->m_lhs(row, col), std::index_sequence_for<args...>{});
            }
        }
    }

    template<class T, T... Is>
    auto apply(real val, std::integer_sequence<T, Is...>) const
    {
        return m_func(val, std::get<Is>(m_args)...);
    }

    unsigned rows() const
    {
        return this->m_lhs.rows();
    }

    unsigned cols() const
    {
        return this->m_lhs.cols();
    }

    unsigned evalCost() const
    {
        // No additional cost
        return 1;
    }

private:
    func m_func;
    std::tuple<args&...> m_args;

};


// Zip
template<class E1, class E2, class func, class... args>
class MatExprZip : public MatExprBase<true, E1, E2>, virtual MatExpr
{
//    using args_seq = std::index_sequence_for<args...>;

public:
    MatExprZip(const E1& lhs, const E2& rhs, func f, const args&... a) :
        MatExprBase<true, E1, E2>(lhs, rhs),
        m_func(f),
        m_args(const_cast<args&>(a)...) {}

    real operator()(const unsigned row, const unsigned col) const
    {
        if (this->m_cached) {
            return this->m_tmp(row, col);
        } else {

            if constexpr (sizeof... (args) == 0) {

                return m_func(this->m_lhs(row, col), this->m_rhs(row, col));
            } else {

                return apply(this->m_lhs(row, col), this->m_rhs(row, col), std::index_sequence_for<args...>{});
            }
        }
    }

    template<class T, T... Is>
    auto apply(real a, real b, std::integer_sequence<T, Is...>) const
    {
        return m_func(a, b, std::get<Is>(m_args)...);
    }

    unsigned rows() const
    {
        return this->m_lhs.rows();
    }

    unsigned cols() const
    {
        return this->m_lhs.cols();
    }

    unsigned evalCost() const
    {
        // No additional cost
        return 1;
    }

private:
    func m_func;
    std::tuple<args&...> m_args;

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
template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
auto operator+(const E1& lhs, const E2& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return MatExprAdd<E1, E2>(lhs, rhs);
}

template <class E1> requires(is_expr<E1>)
auto operator+(const E1& lhs, const real& rhs)
{
    return MatExprAdd<E1, real>(lhs, rhs);
}

template <class E2> requires(is_expr<E2>)
auto operator+(const real& lhs, const E2& rhs)
{
    return MatExprAdd<real, E2>(lhs, rhs);
}


// Subtraction
template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
auto operator-(const E1& lhs, const E2& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return MatExprSub<E1, E2>(lhs, rhs);
}

template <class E1> requires(is_expr<E1>)
auto operator-(const E1& lhs, const real& rhs)
{
    return MatExprSub<E1, real>(lhs, rhs);
}

template <class E2> requires(is_expr<E2>)
auto operator-(const real& lhs, const E2& rhs)
{
    return MatExprSub<real, E2>(lhs, rhs);
}


// Division
template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
auto operator/(const E1& lhs, const E2& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return MatExprDiv<E1, E2>(lhs, rhs);
}

template <class E1> requires(is_expr<E1>)
auto operator/(const E1& lhs, const real& rhs)
{
    return MatExprDiv<E1, real>(lhs, rhs);
}

template <class E2> requires(is_expr<E2>)
auto operator/(const real& lhs, const E2& rhs)
{
    return MatExprDiv<real, E2>(lhs, rhs);
}


// Matrix multiplication
template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
auto operator*(const E1& lhs, const E2& rhs)
{
    assert(lhs.cols() == rhs.rows());
    return MatExprMatMul<E1, E2>(lhs, rhs);
}


// Matrix element wise multiplication, mat ** mat
template <class E1, class E2> requires(is_expr<E1> && is_expr<E2>)
auto operator*(const E1& lhs, const MatExprEWiseMul2<E2>& rhs)
{
    assert(lhs.rows() == rhs.expr.rows());
    assert(lhs.cols() == rhs.expr.cols());
    return MatExprEWiseMul<E1, E2>(lhs, rhs.expr);
}

template <class E2> requires(is_expr<E2>)
auto operator*(const E2& rhs)
{
    return MatExprEWiseMul2<E2>(rhs);
}


// Matrix/Numeric multiplication
template <class E1> requires(is_expr<E1>)
auto operator*(const E1& lhs, const real& rhs)
{
    return MatExprEWiseMul<E1, real>(lhs, rhs);
}

template <class E2> requires(is_expr<E2>)
auto operator*(const real& lhs, const E2& rhs)
{
    return MatExprEWiseMul<real, E2>(lhs, rhs);
}





#endif // MATEXPR_H





















