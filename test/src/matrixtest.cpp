#include "gtest/gtest.h"
#include <NeuralNet/matrix.h>

#define MAT_1_ROWS  4
#define MAT_1_COLS  MAT_1_ROWS
#define MAT_1       {29.9549, -27.7233, 9.8444, 0.0000,     \
                    -14.3950, 0.4060, 1.0000, 23.1001,      \
                    -43.4291, -61.8100, 46.6723, 99.4774,   \
                    -0.2372, 3.5356, 0.0000, 13.7367}       \

#define MAT_2_ROWS  MAT_1_ROWS
#define MAT_2_COLS  MAT_1_ROWS
#define MAT_2       {0.7910, 1.0000, 0.0000, 0.0000,        \
                    -49.8921, 38.0787, 85.0372, -94.1333,   \
                    0.0000, -7.3645, 5.1755, 72.6610,       \
                    -70.6235, -0.5607, 25.5225, 0.2726}     \

#define MAT_3_ROWS  MAT_1_ROWS
#define MAT_3_COLS  3
#define MAT_3       {1.0000, 37.1144, -60.3302,             \
                    48.3185, 0.0000, 67.2406,               \
                    -17.9724, -45.6317, -39.2747,           \
                    0.0000, 11.8273, -0.3022}               \

#define MAT_4_ROWS  MAT_1_ROWS
#define MAT_4_COLS  2
#define MAT_4       {-55.0691, 70.3193,                     \
                    78.9596, 0.0003,                        \
                    0.0000, 1.0007,                         \
                    56.8628, -81.3967}                      \

#define VEC_1_SIZE  MAT_1_ROWS
#define VEC_1       {0.9549, -50.9924, 0.0000, 29.9549}


// Constructors
TEST(MatrixConstructor, Default)
{
    Matrix mat;

    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
}

TEST(MatrixConstructor, RowsCols)
{
    Matrix mat(5, 7);

    EXPECT_EQ(mat.rows(), 5);
    EXPECT_EQ(mat.cols(), 7);
    EXPECT_EQ(mat.vec().size(), 5*7);
}

TEST(MatrixConstructor, RowsColsAndInitVal)
{
    Matrix mat(5, 7, 34.62);

    EXPECT_EQ(mat.rows(), 5);
    EXPECT_EQ(mat.cols(), 7);
    EXPECT_EQ(mat.vec().size(), 5*7);

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), 34.62);
    }
}

TEST(MatrixConstructor, RowsColsAndInitVector)
{
    Vector vec;
    vec = MAT_1;

    Matrix mat(MAT_1_ROWS, MAT_1_COLS, vec);

    EXPECT_EQ(mat.rows(), MAT_1_ROWS);
    EXPECT_EQ(mat.cols(), MAT_1_COLS);
    EXPECT_EQ(mat.vec().size(), MAT_1_ROWS*MAT_1_COLS);

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec(i));
    }
}

TEST(MatrixConstructor, RowsColsAndInitStdVector)
{
    std::vector<real> vec = MAT_1;

    Matrix mat(MAT_1_ROWS, MAT_1_COLS, vec);

    EXPECT_EQ(mat.rows(), MAT_1_ROWS);
    EXPECT_EQ(mat.cols(), MAT_1_COLS);
    EXPECT_EQ(mat.vec().size(), MAT_1_ROWS*MAT_1_COLS);

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec[i]);
    }
}

TEST(MatrixConstructor, CopyMatrix)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    Matrix mat2(mat1);

    EXPECT_EQ(mat1.rows(), mat2.rows());
    EXPECT_EQ(mat1.cols(), mat2.cols());
    EXPECT_EQ(mat1.vec().size(), mat2.vec().size());

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat1(i), mat2(i));
    }
}

TEST(MatrixConstructor, CopyVector)
{
    Vector vec;
    vec = MAT_1;

    Matrix mat(vec);

    EXPECT_EQ(mat.rows(), vec.size());
    EXPECT_EQ(mat.cols(), 1);
    EXPECT_EQ(mat.vec().size(), vec.size());

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec(i));
    }
}

TEST(MatrixConstructor, CopyStdVector)
{
    std::vector<real> vec = MAT_1;

    Matrix mat(vec);

    EXPECT_EQ(mat.rows(), vec.size());
    EXPECT_EQ(mat.cols(), 1);
    EXPECT_EQ(mat.vec().size(), vec.size());

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec[i]);
    }
}

// Assignment operator
TEST(MatrixAssignment, Matrix)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    Matrix mat2;
    mat2 = mat1;


    EXPECT_EQ(mat1.rows(), mat2.rows());
    EXPECT_EQ(mat1.cols(), mat2.cols());
    EXPECT_EQ(mat1.vec().size(), mat2.vec().size());

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat1(i), mat2(i));
    }
}

TEST(MatrixAssignment, Vector)
{
    Vector vec;
    vec = MAT_1;

    Matrix mat;
    mat = vec;

    EXPECT_EQ(mat.rows(), vec.size());
    EXPECT_EQ(mat.cols(), 1);
    EXPECT_EQ(mat.vec().size(), vec.size());

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec(i));
    }
}

TEST(MatrixAssignment, StdVector)
{
    std::vector<real> vec = MAT_1;

    Matrix mat;
    mat = vec;

    EXPECT_EQ(mat.rows(), vec.size());
    EXPECT_EQ(mat.cols(), 1);
    EXPECT_EQ(mat.vec().size(), vec.size());

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec[i]);
    }
}

// Matrix/Matrix operators
TEST(MatrixMatrixOperators, Equallity)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat3(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    ASSERT_TRUE(mat1 == mat2);
    ASSERT_FALSE(mat1 != mat2);
    ASSERT_TRUE(mat1 != mat3);
    ASSERT_FALSE(mat1 == mat3);
}

TEST(MatrixMatrixOperators, Addition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {30.7459, -26.7233, 9.8444, 0.0000,
                               -64.2871, 38.4847, 86.0372, -71.0332,
                               -43.4291, -69.1745, 51.8478, 172.1384,
                               -70.8607, 2.9749, 25.5225, 14.0093});

    EXPECT_TRUE(expected_mat == (mat1 + mat2));
}

TEST(MatrixMatrixOperators, Subtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {29.1639, -28.7233, 9.8444, 0.0000,
                               35.4971, -37.6727, -84.0372, 117.2334,
                               -43.4291, -54.4455, 41.4968, 26.8164,
                               70.3863, 4.0963, -25.5225, 13.4641});

    EXPECT_TRUE(expected_mat == (mat1 - mat2));
}

TEST(MatrixMatrixOperators, Multiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {1406.8680, -1098.2114, -2306.5621, 3324.9897,
                               -1663.0525, -19.2518, 629.2729, 40.7400,
                               -3975.9639, -2796.5687, -2475.6849, 9236.7528,
                               -1146.7200, 126.6917, 651.2525, -329.0731});

    EXPECT_TRUE(expected_mat == mat1 * mat2);
}

TEST(MatrixMatrixOperators, CumulativeAddition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 += mat2;

    EXPECT_TRUE(mat1 == (mat3 + mat2));
}

TEST(MatrixMatrixOperators, CumulativeSubtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 -= mat2;

    EXPECT_TRUE(mat1 == (mat3 - mat2));
}

TEST(MatrixMatrixOperators, CumulativeMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 *= mat2;

    EXPECT_TRUE(mat1 == (mat3 * mat2));
}

// Matrix/Vector operators
TEST(MatrixVectorOperators, Equallity)
{
    Matrix mat(MAT_1_ROWS*MAT_1_COLS, 1, MAT_1);
    Vector vec1(MAT_1);
    Vector vec2(MAT_2);

    ASSERT_TRUE(mat == vec1);
    ASSERT_FALSE(mat != vec1);
    ASSERT_TRUE(mat != vec2);
    ASSERT_FALSE(mat == vec2);
}

TEST(MatrixVectorOperators, Multiplication)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Vector expected_vec({1442.2815, 657.5125, 6090.2054, 230.9662});

    EXPECT_TRUE(expected_vec == (mat * vec));
}

TEST(MatrixVectorOperators, RowWiseAddition)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {30.9098, -26.7684, 10.7993, 0.9549,
                                                 -65.3874, -50.5864, -49.9924, -27.8923,
                                                 -43.4291, -61.8100, 46.6723, 99.4774,
                                                 29.7177, 33.4905, 29.9549, 43.6916});

    EXPECT_TRUE(expected_mat == mat.addRowWise(vec));
}

TEST(MatrixVectorOperators, ColWiseAddition)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {30.9098, -78.7157, 9.8444, 29.9549,
                                                 -13.4401, -50.5864, 1.0000, 53.0550,
                                                 -42.4742, -112.8024, 46.6723, 129.4323,
                                                 0.7177, -47.4568, 0.0000, 43.6916});

    EXPECT_TRUE(expected_mat == mat.addColWise(vec));
}

TEST(MatrixVectorOperators, RowWiseSubtraction)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {29.0000, -28.6782, 8.8895, -0.9549,
                                                 36.5974 ,51.3984, 51.9924, 74.0925,
                                                 -43.4291, -61.8100, 46.6723, 99.4774,
                                                 -30.1921, -26.4193, -29.9549, -16.2182});

    EXPECT_TRUE(expected_mat == mat.subtractRowWise(vec));
}

TEST(MatrixVectorOperators, ColWiseSubtraction)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {29.0000, 23.2691, 9.8444, -29.9549,
                                                 -15.3499, 51.3984, 1.0000, -6.8548,
                                                 -44.3840, -10.8176, 46.6723, 69.5225,
                                                 -1.1921, 54.5280, 0.0000, -16.2182});

    EXPECT_TRUE(expected_mat == mat.subtractColWise(vec));
}

TEST(MatrixVectorOperators, RowWiseMultiplication)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {28.6039, -26.4730, 9.4004, 0.0000,
                                                 734.0356, -20.7029, -50.9924, -1177.9295,
                                                 -0.0000, -0.0000, 0.0000, 0.0000,
                                                 -7.1053, 105.9085, 0.0000, 411.4815});

    EXPECT_TRUE(expected_mat == mat.multiplyRowWise(vec));
}

TEST(MatrixVectorOperators, ColWiseMultiplication)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {28.6039, 1413.6776, 0.0000, 0.0000,
                                                 -13.7458, -20.7029, 0.0000, 691.9612,
                                                 -41.4704, 3151.8402, 0.0000, 2979.8356,
                                                 -0.2265, -180.2887, 0.0000, 411.4815});

    EXPECT_TRUE(expected_mat == mat.multiplyColWise(vec));
}

TEST(MatrixVectorOperators, RowWiseDivision)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);
    vec(2) = 1.0000; // Avoid division with zero

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {31.3697, -29.0327, 10.3094, 0.0000,
                                                 0.2823, -0.0080, -0.0196, -0.4530,
                                                 -43.4291, -61.8100, 46.6723, 99.4774,
                                                 -0.0079, 0.1180, 0.0000, 0.4586});

    EXPECT_TRUE(expected_mat == mat.divideRowWise(vec));
}

TEST(MatrixVectorOperators, ColWiseDivision)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);
    vec(2) = 1.0000; // Avoid division with zero

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {31.3697, 0.5437, 9.8444, 0.0000,
                                                 -15.0749, -0.0080, 1.0000, 0.7712,
                                                 -45.4803, 1.2121, 46.6723, 3.3209,
                                                 -0.2484, -0.0693, 0.0000, 0.4586});

    EXPECT_TRUE(expected_mat == mat.divideColWise(vec));
}

// Matrix/std::vector operators
TEST(MatrixStdVectorOperators, Equallity)
{
    Matrix mat1(MAT_1_ROWS*MAT_1_COLS, 1, MAT_1);
    std::vector<real> vec1(MAT_1);
    std::vector<real> vec2(MAT_2);

    ASSERT_TRUE(mat1 == vec1);
    ASSERT_FALSE(mat1 != vec1);
    ASSERT_TRUE(mat1 != vec2);
    ASSERT_FALSE(mat1 == vec2);
}

TEST(MatrixStdVectorOperators, Multiplication)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    std::vector<real> vec(VEC_1);

    Vector expected_vec({1442.2815, 657.5125, 6090.2054, 230.9662});

    EXPECT_TRUE(expected_vec == (mat * vec));
}

// Matrix/scalar operators
TEST(MatrixScalarOperators, Addition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1 + val;

    for (unsigned i = 0; i < mat2.size(); ++i) {
        EXPECT_FLOAT_EQ(mat2(i), mat1(i) + val);
    }
}

TEST(MatrixScalarOperators, Subtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1 - val;

    for (unsigned i = 0; i < mat2.size(); ++i) {
        EXPECT_FLOAT_EQ(mat2(i), mat1(i) - val);
    }
}

TEST(MatrixScalarOperators, Multiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1 * val;

    for (unsigned i = 0; i < mat2.size(); ++i) {
        EXPECT_FLOAT_EQ(mat2(i), mat1(i) * val);
    }
}

TEST(MatrixScalarOperators, Division)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1 / val;

    for (unsigned i = 0; i < mat2.size(); ++i) {
        EXPECT_FLOAT_EQ(mat2(i), mat1(i) / val);
    }
}

TEST(MatrixScalarOperators, CumulativeAddition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 += val;

    EXPECT_TRUE(mat1 == mat2 + val);
}

TEST(MatrixScalarOperators, CumulativeSubtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 -= val;

    EXPECT_TRUE(mat1 == mat2 - val);
}

TEST(MatrixScalarOperators, CumulativeMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 *= val;

    EXPECT_TRUE(mat1 == mat2 * val);
}

TEST(MatrixScalarOperators, CumulativeDivision)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 /= val;

    EXPECT_TRUE(mat1 == mat2 / val);
}

// Matrix operators
TEST(MatrixOperators, Transpose)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {29.9549, -14.3950, -43.4291, -0.2372,
                                                -27.7233, 0.4060, -61.8100, 3.5356,
                                                9.8444, 1.0000, 46.6723, 0.0000,
                                                0.0000, 23.1001, 99.4774, 13.7367});

    EXPECT_TRUE(expected_mat == mat.transpose());
}

TEST(MatrixOperators, ElementWiseMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1.multiplyElemWise(mat2);

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat3(i), mat1(i) * mat2(i));
    }
}










