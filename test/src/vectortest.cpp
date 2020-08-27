#include "gtest/gtest.h"
#include <NeuralNet/vector.h>
#include <NeuralNet/matrix.h>

// Constructors
TEST(VectorConstructor, Default)
{
    Vector vec;

    EXPECT_EQ(vec.size(), 0);
}

TEST(VectorConstructor, Size)
{
    Vector vec(7);

    EXPECT_EQ(vec.size(), 7);
    EXPECT_EQ(vec.vec().size(), 7);
}

TEST(VectorConstructor, SizeAndInitVal)
{
    Vector vec(7, 3.14);

    EXPECT_EQ(vec.size(), 7);
    EXPECT_EQ(vec.vec().size(), 7);

    for (unsigned i = 0; i < vec.size(); ++i) {
        EXPECT_FLOAT_EQ(vec(i), 3.14);
    }
}

TEST(VectorConstructor, CopyVector)
{
    Vector vec1(7, 3.14);

    Vector vec2(vec1);

    EXPECT_EQ(vec2.size(), vec1.size());
    EXPECT_EQ(vec2.vec().size(), vec1.vec().size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i));
    }
}

TEST(VectorConstructor, CopyMatrix)
{
    Matrix mat(3, 3, 6.22);

    Vector vec(mat);

    EXPECT_EQ(vec.size(), mat.size());
    EXPECT_EQ(vec.vec().size(), mat.vec().size());

    for (unsigned i = 0; i < vec.size(); ++i) {
        EXPECT_FLOAT_EQ(vec(i), mat(i));
    }
}

TEST(VectorConstructor, CopyStdVector)
{
    std::vector<real> vec1;
    vec1.resize(7, 4.56);

    Vector vec2(vec1);

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1[i]);
    }
}

// Assignment operators
TEST(VectorAssignment, Vector)
{
    Vector vec1(7, 3.14);

    Vector vec2;
    vec2 = vec1;

    EXPECT_EQ(vec2.size(), vec1.size());
    EXPECT_EQ(vec2.vec().size(), vec1.vec().size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i));
    }
}

TEST(VectorAssignment, Matrix)
{
    Matrix mat(3, 3, 6.22);

    Vector vec;
    vec = mat;

    EXPECT_EQ(vec.size(), mat.size());
    EXPECT_EQ(vec.vec().size(), mat.vec().size());

    for (unsigned i = 0; i < vec.size(); ++i) {
        EXPECT_FLOAT_EQ(vec(i), mat(i));
    }
}

TEST(VectorAssignment, StdVector)
{
    std::vector<real> vec1;
    vec1.resize(7, 4.56);

    Vector vec2;
    vec2 = vec1;

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1[i]);
    }
}

// Vector/Vector operations
TEST(VectorVectorOperators, Addition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 + vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) + vec2(i));
    }
}

TEST(VectorVectorOperators, Subtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 - vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) - vec2(i));
    }
}

TEST(VectorVectorOperators, Multiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 * vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) * vec2(i));
    }
}

TEST(VectorVectorOperators, Division)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 / vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) / vec2(i));
    }
}

TEST(VectorVectorOperators, CumulativeAddition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 += vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) + vec2(i));
    }
}

TEST(VectorVectorOperators, CumulativeSubtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 -= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) - vec2(i));
    }
}

TEST(VectorVectorOperators, CumulativeMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 *= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) * vec2(i));
    }
}

TEST(VectorVectorOperators, CumulativeDivision)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 /= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) / vec2(i));
    }
}

TEST(VectorVectorOperators, MatrixMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    Vector vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Matrix mat = vec1.matMul(vec2);

    EXPECT_EQ(mat.rows(), vec1.size());
    EXPECT_EQ(mat.cols(), vec2.size());

    for (unsigned i = 0; i < mat.rows(); ++i) {
        for (unsigned j = 0; j < mat.cols(); ++j) {
            EXPECT_FLOAT_EQ(mat(i, j), vec1(i) * vec2(j));
        }
    }
}

// Vector/std::vector operators
TEST(VectorStdVectorOperators, Addition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 + vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) + vec2[i]);
    }
}

TEST(VectorStdVectorOperators, Subtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 - vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) - vec2[i]);
    }
}

TEST(VectorStdVectorOperators, Multiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 * vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) * vec2[i]);
    }
}

TEST(VectorStdVectorOperators, Division)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1 / vec2;

    EXPECT_EQ(vec3.size(), vec1.size());

    for (unsigned i = 0; i < vec3.size(); ++i) {
        EXPECT_FLOAT_EQ(vec3(i), vec1(i) / vec2[i]);
    }
}

TEST(VectorStdVectorOperators, CumulativeAddition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 += vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) + vec2[i]);
    }
}

TEST(VectorStdVectorOperators, CumulativeSubtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 -= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) - vec2[i]);
    }
}

TEST(VectorStdVectorOperators, CumulativeMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 *= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) * vec2[i]);
    }
}

TEST(VectorStdVectorOperators, CumulativeDivision)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    std::vector<real> vec2;
    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

    Vector vec3 = vec1;

    vec1 /= vec2;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec3(i) / vec2[i]);
    }
}

// Vector/scalar operators
TEST(VectorScalarOperators, Addition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1 + val;

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i) + val);
    }
}

TEST(VectorScalarOperators, Subtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1 - val;

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i) - val);
    }
}

TEST(VectorScalarOperators, Multiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1 * val;

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i) * val);
    }
}

TEST(VectorScalarOperators, Division)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1 / val;

    EXPECT_EQ(vec2.size(), vec1.size());

    for (unsigned i = 0; i < vec2.size(); ++i) {
        EXPECT_FLOAT_EQ(vec2(i), vec1(i) / val);
    }
}

TEST(VectorScalarOperators, CumulativeAddition)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1;

    vec1 += val;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec2(i) + val);
    }
}

TEST(VectorScalarOperators, CumulativeSubtraction)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1;

    vec1 -= val;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec2(i) - val);
    }
}

TEST(VectorScalarOperators, CumulativeMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1;

    vec1 *= val;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec2(i) * val);
    }
}

TEST(VectorScalarOperators, CumulativeDivision)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

    real val = 7.23;

    Vector vec2 = vec1;

    vec1 /= val;

    EXPECT_EQ(vec1.size(), vec2.size());

    for (unsigned i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1(i), vec2(i) / val);
    }
}

// Vector/Matrix operators
TEST(VectorMatrixOperators, MatrixMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517};

    Matrix mat(5, 2, {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290});

    Vector vec2 = vec1 * mat;

    EXPECT_EQ(mat.size(), 10);
    EXPECT_EQ(vec1.size(), mat.rows());
    EXPECT_EQ(vec2.size(), mat.cols());

    for (unsigned i = 0; i < vec2.size(); ++i) {

        real sum = 0;

        for (unsigned j = 0; j < vec1.size(); ++j) {
            sum += vec1(j)*mat(j*mat.cols() + i);
        }

        EXPECT_FLOAT_EQ(vec2(i), sum);
    }
}

TEST(VectorMatrixOperators, CumulativeMatrixMultiplication)
{
    Vector vec1;
    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517};

    Matrix mat(5, 2, {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
            17.1146, 24.1795, 3.9167, 95.6260, 47.6290});

    Vector vec2 = vec1;

    vec2 *= mat;

    EXPECT_EQ(mat.size(), 10);
    EXPECT_EQ(vec1.size(), mat.rows());
    EXPECT_EQ(vec2.size(), mat.cols());

    for (unsigned i = 0; i < vec2.size(); ++i) {

        real sum = 0;

        for (unsigned j = 0; j < vec1.size(); ++j) {
            sum += vec1(j)*mat(j*mat.cols() + i);
        }

        EXPECT_FLOAT_EQ(vec2(i), sum);
    }
}









































