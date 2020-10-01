#ifndef MATRIX_H
#define MATRIX_H

#include <Common/types.h>
#include <vector>
#include <NeuralNet/matexpr_base.h>


#include <thread>
#include <algorithm>

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
        evalExpr(expr);
    }

    // Matrix/Matrix operators
    bool operator==(const Matrix& rhs) const;                                         // vec: exactly same
    bool operator!=(const Matrix& rhs) const;                                         // vec: exactly same
    template <typename E>
    Matrix& operator+=(const MatExpr<E>& rhs)
    {
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

    template <class E>
    auto mulEWise(const MatExpr<E>& rhs)
    {
        return MatExprEWiseMul<Matrix, E>(*this, *static_cast<const E*>(&rhs));
    }

    template <class E1, class E2>
    static auto mulEWise(const MatExpr<E1>& lhs, const MatExpr<E2>& rhs)
    {
        return MatExprEWiseMul<E1, E2>(*static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
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

    template <typename E>
    Matrix operator=(const MatExpr<E>& rhs)
    {
        m_rows = rhs.rows();
        m_cols = rhs.cols();
        m_vec.resize(m_rows*m_cols);

        evalExpr(rhs);

        return *this;
    }

    template <typename E>
    void evalExpr(const MatExpr<E>& rhs)
    {
        // Cache if needed
        const_cast<MatExpr<E>&>(rhs).sourceOk(*this);

        // Eval
        parallel_for([&](unsigned start, unsigned end)
        {
            for (unsigned i = start; i < end; ++i) {
                m_vec[i] = rhs(i / m_cols, i % m_cols);
            }
        }, m_rows*m_cols);
    }



//    template<typename T, typename... Ts>
//    struct contains : std::disjunction<std::is_same<T, Ts>...>
//    {};

//    template<typename T, typename... Ts>
//    struct containsMatExpr : std::disjunction<std::is_same<MatExpr<T>, Ts>...>
//    {};

//    template <typename T, typename... Ts>
//    struct Index;

//    template <typename T, typename... Ts>
//    struct Index<T, T, Ts...> : std::integral_constant<std::size_t, 0> {};

//    template <typename T, typename U, typename... Ts>
//    struct Index<T, U, Ts...> : std::integral_constant<std::size_t, 1 + Index<T, Ts...>::value> {};

//    template <typename E>
//    struct Index<T, U, Ts...> : std::integral_constant<std::size_t, 1 + Index<T, Ts...>::value> {};

//    template<int N, typename... Ts> using NthTypeOf =
//            typename std::tuple_element<N, std::tuple<Ts...>>::type;


//    template<class E>
//    constexpr int ac = Index<MatExpr<E>, args...>::value;

//    using ab = NthTypeOf<ac, args...>;


    // Matrix::apply(net.activFunc, expr, funcArgs);
    // Matrix::apply(net.activFunc, funcArgs, expr);
    // Matrix::apply(net.activFunc, expr);
    template<class func, class E, class... args>
    static auto apply(func f, const MatExpr<E>& expr, const args&... a)
    {        
        return MatExprApply<func, E, args...>(f, *static_cast<const E*>(&expr), a...);
    }

//    template<class func, class... args, class E>
//    static auto apply(func f, const args&... a, const MatExpr<E>& expr)
//    {
//        return MatExprApply<func, E, args...>(f, *static_cast<const E*>(&expr), a...);
//    }

    template<class func, class E>
    static auto apply(func f, const MatExpr<E>& expr)
    {
        return MatExprApply<func, E>(f, *static_cast<const E*>(&expr));
    }

    template<class func, class E1, class E2, class... args>
    static auto zip(func f, const MatExpr<E1>& lhs, const MatExpr<E2>& rhs, const args&... a)
    {
        return MatExprZip<func, E1, E2, args...>(f, *static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs), a...);
    }

//    template<class func, class... args, class E1, class E2>
//    static auto zip(func f, const args&... a, const MatExpr<E>& expr)
//    {
//        return MatExprApply<func, E, args...>(f, *static_cast<const E*>(&expr), a...);
//    }

    template<class func, class E1, class E2>
    static auto zip(func f, const MatExpr<E1>& lhs, const MatExpr<E2>& rhs)
    {
        return MatExprZip<func, E1, E2>(f, *static_cast<const E1*>(&lhs), *static_cast<const E2*>(&rhs));
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


































































