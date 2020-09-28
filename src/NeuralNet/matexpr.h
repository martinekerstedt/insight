#ifndef MATEXPR_H
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


// Binary expressions
// Addition
template <typename E1, typename E2>
class MatExprAdd : RequireOverride, public MatExpr<MatExprAdd<E1, E2> >
{

public:
    MatExprAdd(const E1& lhs, const E2& rhs) :
        m_lhs(lhs),
        m_rhs(rhs)
    {
        if constexpr (!std::is_arithmetic_v<E1> && !std::is_arithmetic_v<E2>) {
            assert(m_lhs.size() == m_rhs.size());
        }
    }

    real operator()(unsigned row, unsigned col) const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return m_lhs + m_rhs(row, col);
        } else if constexpr (std::is_arithmetic_v<E2>) {
            return m_lhs(row, col) + m_rhs;
        } else {
            return m_lhs(row, col) + m_rhs(row, col);
        }
    }

    unsigned rows() const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return m_rhs.rows();
        } else {
            return m_lhs.rows();
        }
    }

    unsigned cols() const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return m_rhs.cols();
        } else {
            return m_lhs.cols();
        }
    }

    bool sourceOk(const Matrix& destMat)
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return const_cast<E2&>(m_rhs).sourceOk(destMat);
        } else if constexpr (std::is_arithmetic_v<E2>) {
            return const_cast<E1&>(m_lhs).sourceOk(destMat);
        } else {
            return const_cast<E1&>(m_lhs).sourceOk(destMat)
                    || const_cast<E2&>(m_rhs).sourceOk(destMat);
        }
    }

    unsigned evalCost() const
    {
        // One operation per element, plus the cost of evaluating the
        // operands
        if constexpr (std::is_arithmetic_v<E1>) {
            return rows()*cols() + m_rhs.evalCost();
        } else if constexpr (std::is_arithmetic_v<E2>) {
            return rows()*cols() + m_lhs.evalCost();
        } else {
            return rows()*cols() + m_lhs.evalCost() + m_rhs.evalCost();
        }
    }

protected:
    const E1& m_lhs;
    const E2& m_rhs;

};


// Subtraction
template <typename E1, typename E2>
struct MatExprSub : public MatExprAdd<E1, E2>
{
    MatExprSub(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return this->m_lhs - this->m_rhs(row, col);
        } else if constexpr (std::is_arithmetic_v<E2>) {
            return this->m_lhs(row, col) - this->m_rhs;
        } else {
            return this->m_lhs(row, col) - this->m_rhs(row, col);
        }
    }
};


// Division, only defined for Mat / Num
template <typename E1, typename E2>
struct MatExprDiv : public MatExprAdd<E1, E2>
{
    MatExprDiv(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs)
    {
        static_assert(std::is_arithmetic_v<E2>, "Only defined for Mat / Num");
    }

    real operator()(unsigned row, unsigned col) const
    {
        return this->m_lhs(row, col) / this->m_rhs;
    }
};


// Element wise multiplication
template <typename E1, typename E2>
struct MatExprEWiseMul : public MatExprAdd<E1, E2>
{
    MatExprEWiseMul(const E1& lhs, const E2& rhs) :
        MatExprAdd<E1, E2>(lhs, rhs) {}

    real operator()(unsigned row, unsigned col) const
    {
        if constexpr (std::is_arithmetic_v<E1>) {
            return this->m_lhs * this->m_rhs(row, col);
        } else if constexpr (std::is_arithmetic_v<E2>) {
            return this->m_lhs(row, col) * this->m_rhs;
        } else {
            return this->m_lhs(row, col) * this->m_rhs(row, col);
        }
    }
};


// Martrix multiplication
template <typename E1, typename E2>
class MatExprMatMul : RequireOverride, public MatExpr<MatExprMatMul<E1, E2> >
{

public:
    MatExprMatMul(const E1& lhs, const E2& rhs) :
        m_lhs(lhs),
        m_rhs(rhs),
        m_cached(false)
    {
        static_assert (!std::is_arithmetic_v<E1> && !std::is_arithmetic_v<E2>, "Must be MatExpr");
        assert(lhs.cols() == rhs.rows());
    }

    real operator()(unsigned row, unsigned col) const
    {
        if (m_cached) {
            return m_tmp(row, col);
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
        if (!const_cast<E1&>(m_lhs).sourceOk(destMat)
                || !const_cast<E2&>(m_rhs).sourceOk(destMat)) {

            m_tmp.resize(rows(), cols());

            for (unsigned row = 0; row < rows(); ++row) {
                for (unsigned col = 0; col < cols(); ++col) {
                    m_tmp(row, col) = operator()(row, col);
                }
            }

            m_cached = true;
        }

        return true;
    }

    unsigned evalCost() const
    {
        return rows()*cols()*m_lhs.cols() + m_lhs.evalCost() + m_rhs.evalCost();
    }

private:
    const E1& m_lhs;
    const E2& m_rhs;
    Matrix m_tmp;
    bool m_cached;

};




// Unary expressions
// Transposition
template <typename E>
class MatExprTrans : RequireOverride, public MatExpr<MatExprTrans<E> >
{

public:
    MatExprTrans(const E& expr) :
        m_expr(expr),
        m_cached(false)
    {
        static_assert (!std::is_arithmetic_v<E>, "Must be MatExpr");
    }

    real operator()(const unsigned row, const unsigned col) const
    {
        if (m_cached) {
            return m_tmp(row, col);
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

    bool sourceOk(const Matrix& destMat)
    {
        if (!const_cast<E&>(m_expr).sourceOk(destMat)) {
            m_tmp.resize(rows(), cols());

            for (unsigned row = 0; row < rows(); ++row) {
                for (unsigned col = 0; col < cols(); ++col) {
                    m_tmp(row, col) = operator()(row, col);
                }
            }

            m_cached = true;
        }

        return true;
    }

    unsigned evalCost() const
    {
        // No additional cost
        return m_expr.evalCost();
    }

private:
    const E& m_expr;
    Matrix m_tmp;
    bool m_cached;

};




// Apply
template <typename E>
class MatExprApply : RequireOverride, public MatExpr<MatExprApply<E> >
{

public:
    MatExprApply(real (*func)(real), const E& expr) :
        m_func(func),
        m_expr(expr)
    {
        static_assert (!std::is_arithmetic_v<E>, "Must be MatExpr");
    }

    real operator()(const unsigned row, const unsigned col) const
    {
        return m_func(m_expr(col, row));
    }

    unsigned rows() const
    {
        return m_expr.rows();
    }

    unsigned cols() const
    {
        return m_expr.cols();
    }

    bool sourceOk(const Matrix& destMat)
    {
        return const_cast<E&>(m_expr).sourceOk(destMat);
    }

    unsigned evalCost() const
    {
        // No additional cost
        return m_expr.evalCost();
    }

private:
    real (*m_func)(real);
    const E& m_expr;

};











template <class T> struct is_array {
    static constexpr bool value = false; };

template <class T>
struct is_array<std::vector<T>> {
    static constexpr bool value = true;
};

template <class T>
constexpr bool is_array_v =
        is_array<std::remove_cvref_t<T>>::value;

struct expression {};

//template <class callable, class... operands>
//class expr : public expression { /* ... */ };

template <class T>
constexpr bool is_array_or_expression =
        is_array_v<T> ||
        std::is_base_of_v<expression, std::remove_cvref_t<T>>;
template <class A, class B>

constexpr bool is_binary_op_ok =
        is_array_or_expression<A> ||
        is_array_or_expression<B>;


template <class operand>
auto subscript(operand const& v, size_t const i) {
    if constexpr (is_array_or_expression<operand>) {
        return v[i];
    } else {
        return v;
    }
}

#include <tuple>

template <class callable, class... operands>
class expr
{

public:
    expr(callable f, operands const&... args) :
        args_(args...),
        f_(f)
    {

    }

    auto operator[](size_t const i) const
    {
        auto const call_at_index =
                [this, i](operands const&... a) {
            return f_(subscript(a, i)...);
        };

        return std::apply(call_at_index, args_);
    }

private:
    std::tuple<operands const&...> args_;
    callable f_;

};

class tridiagonal
{

public:
    template <class src_type>
    tridiagonal& operator=(src_type const& src)
    {
        size_t const I = v_.size();

        for (size_t i = 0; i < I; ++i) {
            v_[i] = src[i];
        }

        return *this;
    }

private:
    std::vector<double> v_;

};

template <class LHS, class RHS> requires(is_binary_op_ok<LHS, RHS>)
auto operator*(LHS const& lhs, RHS const& rhs)
{
    return expr{
        [](auto const& l, auto const& r) {
            return l * r; },
        lhs, rhs};
}





// Operators
//
// The template signature "typename = std::enable_if_t<std::is_arithmetic_v<T>>"
// conditionally removes the function declaration if T is not an arithmetic type
//
// Addition
template <typename E1, typename E2>
auto operator+(const MatExpr<E1>& lhs, const MatExpr<E2>& rhs)
{
    return MatExprAdd<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
}

template <typename E1>
auto operator+(const MatExpr<E1>& lhs, const real& rhs)
{
    return MatExprAdd<E1, real>(*static_cast<const E1*>(&lhs), rhs);
}

template <typename E2>
auto operator+(const real& lhs, const MatExpr<E2>& rhs)
{
    return MatExprAdd<real, E2>(lhs, *static_cast<const E2*>(&rhs));
}


// Subtraction, not defined for Num - Mat
template <typename E1, typename E2>
auto operator-(const MatExpr<E1>& lhs, const MatExpr<E2>& rhs)
{
    return MatExprSub<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
}

template <typename E1>
auto operator-(const MatExpr<E1>& lhs, const real& rhs)
{
    return MatExprSub<E1, real>(*static_cast<const E1*>(&lhs), rhs);
}


// Division, only defined for Mat / Num
template <typename E1>
auto operator/(const MatExpr<E1>& lhs, const real& rhs)
{
    return MatExprDiv<E1, real>(*static_cast<const E1*>(&lhs), rhs);
}


// Matrix multiplication
template <typename E1, typename E2>
auto operator*(const MatExpr<E1>& lhs, const MatExpr<E2>& rhs)
{
    return MatExprMatMul<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
}


// Matrix/Numeric multiplication
template <typename E1>
auto operator*(const MatExpr<E1>& lhs, const real& rhs)
{
    return MatExprEWiseMul<E1, real>(*static_cast<const E1*>(&lhs), rhs);
}

template <typename E2>
auto operator*(const real& lhs, const MatExpr<E2>& rhs)
{
    return MatExprEWiseMul<real, E2>(lhs, *static_cast<const E2*>(&rhs));
}

template <typename E>
auto apply(real (*func)(real), const MatExpr<E>& expr)
{
    return MatExprApply<E>(func, *static_cast<E*>(&expr));
}

#endif // MATEXPR_H





















