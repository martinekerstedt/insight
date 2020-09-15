#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <stdexcept>
#include <sstream>
#include <limits>

using real = float;
using real_vec = std::vector<real>;
using real_matrix = std::vector<real_vec>;
using real_3d_matrix = std::vector<real_matrix>;

#define EPSILON 0.001f

#define THROW_ERROR(X) std::stringstream ss; \
                        ss << X; \
                        throw std::invalid_argument(std::string(std::string(__FILE__) \
                        + " | func:" + std::string(__FUNCTION__) \
                        + " | line:" + std::to_string(__LINE__) \
                        + " > " + ss.str()))

#define NONE_UL     std::numeric_limits<unsigned long>::quiet_NaN()
#define NONE_REAL   std::numeric_limits<real>::quiet_NaN()

namespace InitializationFunction {
    struct ALL_ZERO
    {

    };

    struct RANDOM_NORMAL
    {
        real mean = 0.0;
        real stddev = 1.0;
        unsigned long seed = NONE_UL;
    };
}

namespace CostFunction {
    struct DIFFERENCE
    {

    };

    struct CROSS_ENTROPY
    {

    };

    struct SQUARE_DIFFERENCE
    {

    };
}

namespace ActivationFunction {
    struct RELU
    {
        real alpha = 0.0;
        real max_value = NONE_REAL;
        real threshold = 0.0;
    };

    struct SIGMOID
    {

    };

    struct TANH
    {

    };
}

namespace OptimizeFunction {
    struct TEST
    {

    };

    struct BACKPROP
    {
        real learningRate = 0.01;
        real momentum = 0.0;
    };
}





//namespace base {

//template <typename F>
//struct named_operator_wrapper {
//    F f;
//};

//template <typename T, typename F>
//struct named_operator_lhs {
//    F f;
//    T& value;
//};

//template <typename T, typename F>
//inline named_operator_lhs<T, F> operator <(T& lhs, named_operator_wrapper<F> rhs) {
//    return {rhs.f, lhs};
//}

//template <typename T, typename F>
//inline named_operator_lhs<T const, F> operator <(T const& lhs, named_operator_wrapper<F> rhs) {
//    return {rhs.f, lhs};
//}

//template <typename T1, typename T2, typename F>
//inline auto operator >(named_operator_lhs<T1, F> const& lhs, T2 const& rhs)
//    -> decltype(lhs.f(std::declval<T1>(), std::declval<T2>()))
//{
//    return lhs.f(lhs.value, rhs);
//}

//template <typename T1, typename T2, typename F>
//inline auto operator >=(named_operator_lhs<T1, F> const& lhs, T2 const& rhs)
//    -> decltype(lhs.value = lhs.f(std::declval<T1>(), std::declval<T2>()))
//{
//    return lhs.value = lhs.f(lhs.value, rhs);
//}

//template <typename F>
//inline constexpr named_operator_wrapper<F> make_named_operator(F f) {
//    return {f};
//}

//} // namespace base

//namespace op {
//    std::pair<int, int> divmod(int x, int y) {
//        return { x / y, x % y };
//    }

//    struct append {
//        template <typename T>
//        std::vector<T> operator ()(std::vector<T> const& vs, T const& v) const {
//            auto copy(vs);
//            copy.push_back(v);
//            return copy;
//        }
//    };
//} // namespace op


#endif // TYPES_H
