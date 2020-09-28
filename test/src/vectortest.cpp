#include "gtest/gtest.h"
#include <NeuralNet/vector.h>
//#include <NeuralNet/matrix.h>

#define MAT_1_ROWS  4
#define MAT_1_COLS  MAT_1_ROWS
#define MAT_1       {29.9549, -27.7233, 9.8444, 0.0000,     \
                    -14.3950, 0.4060, 1.0000, 23.1001,      \
                    -43.4291, -61.8100, 46.6723, 99.4774,   \
                    -0.2372, 3.5356, 0.0000, 13.7367}

#define MAT_2_ROWS  MAT_1_ROWS
#define MAT_2_COLS  MAT_1_ROWS
#define MAT_2       {0.7910, 1.0000, 0.0000, 0.0000,        \
                    -49.8921, 38.0787, 85.0372, -94.1333,   \
                    0.0000, -7.3645, 5.1755, 72.6610,       \
                    -70.6235, -0.5607, 25.5225, 0.2726}

#define MAT_3_ROWS  MAT_1_ROWS
#define MAT_3_COLS  3
#define MAT_3       {1.0000, 37.1144, -60.3302,             \
                    48.3185, 0.0000, 67.2406,               \
                    -17.9724, -45.6317, -39.2747,           \
                    0.0000, 11.8273, -0.3022}

#define MAT_4_ROWS  1
#define MAT_4_COLS  MAT_1_ROWS
#define MAT_4       {-55.0691, 70.3193, 78.9596, 0.0003}

#define VEC_1_SIZE  MAT_1_ROWS
#define VEC_1       {0.9549,                                \
                    -50.9924,                               \
                    0.0000,                                 \
                    29.9549}

testing::AssertionResult equal_matricies(Matrix mat1, Matrix mat2);

//testing::AssertionResult equal_matricies(Matrix mat1, Matrix mat2)
//{
//    std::string mat1_str = mat1.str();
//    std::string mat2_str = mat2.str();


//    if (mat1_str == mat2_str) {
//        return testing::AssertionSuccess();
//    } else {
//        return testing::AssertionFailure() << "\n\nExpected:\n" << mat1_str
//                                           << "\nBut got:\n" << mat2_str
//                                           << std::endl;
//    }
//}

// Constructors
TEST(VectorConstructor, Default)
{
    Vector vec;

    EXPECT_EQ(vec.rows(), 0);
    EXPECT_EQ(vec.cols(), 0);
}

TEST(VectorConstructor, Size)
{
    Vector vec(7);

    EXPECT_EQ(vec.rows(), 7);
    EXPECT_EQ(vec.cols(), 1);
    EXPECT_EQ(vec.vec().size(), 7);
}

TEST(VectorConstructor, SizeInitVal)
{
    Vector vec(7, 34.62);

    EXPECT_EQ(vec.rows(), 7);
    EXPECT_EQ(vec.cols(), 1);
    EXPECT_EQ(vec.vec().size(), 7);

    for (unsigned i = 0; i < vec.size(); ++i) {
        EXPECT_FLOAT_EQ(vec(i), 34.62);
    }
}

// Copy constructor
TEST(VectorCopyConstructor, Vector)
{
    Vector vec1(VEC_1);
    Vector vec2(vec1);

    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

TEST(VectorCopyConstructor, Matrix)
{
    Matrix mat(MAT_1);
    Vector vec(mat);

    EXPECT_TRUE(equal_matricies(vec, mat));
}

TEST(VectorCopyConstructor, StdVector)
{
    std::vector<real> vec1(MAT_1);
    Vector vec2(vec1);

    // vec == mat not possible, cannot edit std::vector class
    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

TEST(VectorCopyConstructor, InitializerList)
{
    std::vector<real> vec1(MAT_1);
    Vector vec2(MAT_1);

    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

// Assignment operator
TEST(VectorAssignment, Vector)
{
    Vector vec1(MAT_1);

    Vector vec2;
    vec2 = vec1;

    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

TEST(VectorAssignment, Matrix)
{
    Matrix mat(MAT_1);

    Vector vec;
    vec = mat;

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(VectorAssignment, StdVector)
{
    std::vector<real> vec1(MAT_1);

    Vector vec2;
    vec2 = vec1;

    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

TEST(VectorAssignment, InitializerList)
{
    std::vector<real> vec1(MAT_1);
    Vector vec2;
    vec2 = MAT_1;

    EXPECT_TRUE(equal_matricies(vec1, vec2));
}

// Vector functions
TEST(VectorFunctions, Transpose)
{
    Vector vec1(VEC_1);
    Vector vec2 = vec1.trans();

    EXPECT_EQ(vec1.rows(), VEC_1_SIZE);
    EXPECT_EQ(vec1.cols(), 1);
    EXPECT_EQ(vec2.rows(), 1);
    EXPECT_EQ(vec2.cols(), VEC_1_SIZE);
}







































//// Constructors
//TEST(VectorConstructor, Default)
//{
//    Vector vec;

//    EXPECT_EQ(vec.size(), 0);
//}

//TEST(VectorConstructor, Size)
//{
//    Vector vec(7);

//    EXPECT_EQ(vec.size(), 7);
//    EXPECT_EQ(vec.vec().size(), 7);
//}

//TEST(VectorConstructor, SizeAndInitVal)
//{
//    Vector vec(7, 3.14);

//    EXPECT_EQ(vec.size(), 7);
//    EXPECT_EQ(vec.vec().size(), 7);

//    for (unsigned i = 0; i < vec.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec(i), 3.14);
//    }
//}

//TEST(VectorConstructor, CopyVector)
//{
//    Vector vec1(7, 3.14);

//    Vector vec2(vec1);

//    EXPECT_EQ(vec2.size(), vec1.size());
//    EXPECT_EQ(vec2.vec().size(), vec1.vec().size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i));
//    }
//}

//TEST(VectorConstructor, CopyMatrix)
//{
//    Matrix mat(3, 3, 6.22);

//    Vector vec(mat);

//    EXPECT_EQ(vec.size(), mat.size());
//    EXPECT_EQ(vec.vec().size(), mat.vec().size());

//    for (unsigned i = 0; i < vec.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec(i), mat(i));
//    }
//}

//TEST(VectorConstructor, CopyStdVector)
//{
//    std::vector<real> vec1;
//    vec1.resize(7, 4.56);

//    Vector vec2(vec1);

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1[i]);
//    }
//}

//// Assignment operators
//TEST(VectorAssignment, Vector)
//{
//    Vector vec1(7, 3.14);

//    Vector vec2;
//    vec2 = vec1;

//    EXPECT_EQ(vec2.size(), vec1.size());
//    EXPECT_EQ(vec2.vec().size(), vec1.vec().size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i));
//    }
//}

//TEST(VectorAssignment, Matrix)
//{
//    Matrix mat(3, 3, 6.22);

//    Vector vec;
//    vec = mat;

//    EXPECT_EQ(vec.size(), mat.size());
//    EXPECT_EQ(vec.vec().size(), mat.vec().size());

//    for (unsigned i = 0; i < vec.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec(i), mat(i));
//    }
//}

//TEST(VectorAssignment, StdVector)
//{
//    std::vector<real> vec1;
//    vec1.resize(7, 4.56);

//    Vector vec2;
//    vec2 = vec1;

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1[i]);
//    }
//}

//// Vector/Vector operations
//TEST(VectorVectorOperators, Addition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 + vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) + vec2(i));
//    }
//}

//TEST(VectorVectorOperators, Subtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 - vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) - vec2(i));
//    }
//}

//TEST(VectorVectorOperators, Multiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 * vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) * vec2(i));
//    }
//}

//TEST(VectorVectorOperators, Division)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 / vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) / vec2(i));
//    }
//}

//TEST(VectorVectorOperators, CumulativeAddition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 += vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) + vec2(i));
//    }
//}

//TEST(VectorVectorOperators, CumulativeSubtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 -= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) - vec2(i));
//    }
//}

//TEST(VectorVectorOperators, CumulativeMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 *= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) * vec2(i));
//    }
//}

//TEST(VectorVectorOperators, CumulativeDivision)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 /= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) / vec2(i));
//    }
//}

//TEST(VectorVectorOperators, MatrixMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    Vector vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Matrix mat = vec1.matMul(vec2);

//    EXPECT_EQ(mat.rows(), vec1.size());
//    EXPECT_EQ(mat.cols(), vec2.size());

//    for (unsigned i = 0; i < mat.rows(); ++i) {
//        for (unsigned j = 0; j < mat.cols(); ++j) {
//            EXPECT_FLOAT_EQ(mat(i, j), vec1(i) * vec2(j));
//        }
//    }
//}

//// Vector/std::vector operators
//TEST(VectorStdVectorOperators, Addition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 + vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) + vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, Subtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 - vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) - vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, Multiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 * vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) * vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, Division)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1 / vec2;

//    EXPECT_EQ(vec3.size(), vec1.size());

//    for (unsigned i = 0; i < vec3.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec3(i), vec1(i) / vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, CumulativeAddition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 += vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) + vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, CumulativeSubtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 -= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) - vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, CumulativeMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 *= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) * vec2[i]);
//    }
//}

//TEST(VectorStdVectorOperators, CumulativeDivision)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    std::vector<real> vec2;
//    vec2 = {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290};

//    Vector vec3 = vec1;

//    vec1 /= vec2;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec3(i) / vec2[i]);
//    }
//}

//// Vector/scalar operators
//TEST(VectorScalarOperators, Addition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1 + val;

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i) + val);
//    }
//}

//TEST(VectorScalarOperators, Subtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1 - val;

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i) - val);
//    }
//}

//TEST(VectorScalarOperators, Multiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1 * val;

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i) * val);
//    }
//}

//TEST(VectorScalarOperators, Division)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1 / val;

//    EXPECT_EQ(vec2.size(), vec1.size());

//    for (unsigned i = 0; i < vec2.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec2(i), vec1(i) / val);
//    }
//}

//TEST(VectorScalarOperators, CumulativeAddition)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1;

//    vec1 += val;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec2(i) + val);
//    }
//}

//TEST(VectorScalarOperators, CumulativeSubtraction)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1;

//    vec1 -= val;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec2(i) - val);
//    }
//}

//TEST(VectorScalarOperators, CumulativeMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1;

//    vec1 *= val;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec2(i) * val);
//    }
//}

//TEST(VectorScalarOperators, CumulativeDivision)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517,
//            22.0866, 10.2207, 84.6985, 45.7489, 56.1249};

//    real val = 7.23;

//    Vector vec2 = vec1;

//    vec1 /= val;

//    EXPECT_EQ(vec1.size(), vec2.size());

//    for (unsigned i = 0; i < vec1.size(); ++i) {
//        EXPECT_FLOAT_EQ(vec1(i), vec2(i) / val);
//    }
//}

//// Vector/Matrix operators
//TEST(VectorMatrixOperators, MatrixMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517};

//    Matrix mat(5, 2, {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290});

//    Vector vec2 = vec1 * mat;

//    EXPECT_EQ(mat.size(), 10);
//    EXPECT_EQ(vec1.size(), mat.rows());
//    EXPECT_EQ(vec2.size(), mat.cols());

//    for (unsigned i = 0; i < vec2.size(); ++i) {

//        real sum = 0;

//        for (unsigned j = 0; j < vec1.size(); ++j) {
//            sum += vec1(j)*mat(j*mat.cols() + i);
//        }

//        EXPECT_FLOAT_EQ(vec2(i), sum);
//    }
//}

//TEST(VectorMatrixOperators, CumulativeMatrixMultiplication)
//{
//    Vector vec1;
//    vec1 = {4.7623, 79.4754, 38.1879, 49.5188, 75.7517};

//    Matrix mat(5, 2, {28.2063, 5.7376, 75.3608, 83.5950, 63.7761,
//            17.1146, 24.1795, 3.9167, 95.6260, 47.6290});

//    Vector vec2 = vec1;

//    vec2 *= mat;

//    EXPECT_EQ(mat.size(), 10);
//    EXPECT_EQ(vec1.size(), mat.rows());
//    EXPECT_EQ(vec2.size(), mat.cols());

//    for (unsigned i = 0; i < vec2.size(); ++i) {

//        real sum = 0;

//        for (unsigned j = 0; j < vec1.size(); ++j) {
//            sum += vec1(j)*mat(j*mat.cols() + i);
//        }

//        EXPECT_FLOAT_EQ(vec2(i), sum);
//    }
//}









































