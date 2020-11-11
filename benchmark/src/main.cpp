#include <benchmark/benchmark.h>
#include <Matrix/matrix.h>

//
// https://github.com/google/benchmark/issues/242
//
// DoNotOptimize(<expr>) works by forcing the result of <expr> to be stored to memory,
// which in turn forces the compiler toactually evaluate <expr>. It does not prevent the
// compiler from optimizing the evaluation of <expr> but it does prevent the expression
// from being discarded completely.
//
// As you noted in your example the compiler optimized <expr> so that it only had to
// be evaluated once and therefore could reuse the result each loop iteration.
// Unfortunately you just have to be aware of these Gotcha's when writing benchmarks.
//
// In my experience it's important to give the benchmark different inputs on every
// iteration to prevent this kind of optimization from taking place.
//

#include <random>
#include <iostream>

Matrix randomMatrix(unsigned rows, unsigned cols)
{
    Matrix mat(rows, cols);

    std::mt19937 gen;
    std::random_device rd{};
    gen.seed(rd());
    std::normal_distribution<real> d{0, 3};

    for (unsigned i = 0; i < mat.size(); ++i) {
        mat(i) = d(gen);
    }

    return mat;
}

static void MatrixTranspose(benchmark::State& state)
{
    unsigned a = std::pow(2, state.range(0));
    Matrix mat = randomMatrix(a, a);
    Matrix res_mat(a, a);

//    std::cout << "Matrix size: " << a << "x" << a << " = " << a*a << std::endl;

    for (auto _ : state) {
        benchmark::DoNotOptimize(res_mat = mat.trans());
    }

}

BENCHMARK(MatrixTranspose)->DenseRange(1, 11, 1);




































