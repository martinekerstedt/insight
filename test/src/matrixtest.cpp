#include "gtest/gtest.h"
#include <Matrix/matrix.h>
#include <Matrix/vector.h>

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


testing::AssertionResult equal_matricies(Matrix mat1, Matrix mat2)
{    
    std::string mat1_str = mat1.str();
    std::string mat2_str = mat2.str();

    if (mat1.rows() != mat2.rows()) {
        return testing::AssertionFailure() << "\n\nExpected:\n" << mat1_str
                                           << "\nBut got:\n" << mat2_str
                                           << std::endl;
    }

    if (mat1.cols() != mat2.cols()) {
        return testing::AssertionFailure() << "\n\nExpected:\n" << mat1_str
                                           << "\nBut got:\n" << mat2_str
                                           << std::endl;
    }

    for (unsigned i = 0; i < mat1.size(); ++i) {
        const ::testing::internal::FloatingPoint<real> lhs(mat1(i)), rhs(mat2(i));

        if (!lhs.AlmostEquals(rhs)) {
            return testing::AssertionFailure() << "\n\n\nExpected:\n" << mat1_str
                                               << "\nBut got:\n" << mat2_str
                                               << "\nError at (" << i / mat1.cols() << ", " << i % mat1.cols() << ")\n"
                                               << "Expected: " << mat1(i) << "\n"
                                               << "But got: " << mat2(i) << "\n\n"
                                               << std::endl;
        }
    }

    return testing::AssertionSuccess();
}


// Constructors
TEST(MatrixConstructor, Default)
{
    Matrix mat;

    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
    EXPECT_EQ(mat.vec().size(), 0);
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
    Vector vec(MAT_1);
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
    std::vector<real> vec(MAT_1);
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, vec);

    EXPECT_EQ(mat.rows(), MAT_1_ROWS);
    EXPECT_EQ(mat.cols(), MAT_1_COLS);
    EXPECT_EQ(mat.vec().size(), MAT_1_ROWS*MAT_1_COLS);

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec[i]);
    }
}

TEST(MatrixConstructor, RowsColsAndInitializerList)
{
    std::vector<real> vec(MAT_1);
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    EXPECT_EQ(mat.rows(), MAT_1_ROWS);
    EXPECT_EQ(mat.cols(), MAT_1_COLS);
    EXPECT_EQ(mat.vec().size(), MAT_1_ROWS*MAT_1_COLS);

    for (unsigned i = 0; i < mat.size(); ++i) {
        EXPECT_FLOAT_EQ(mat(i), vec[i]);
    }
}

// Copy constructor
TEST(MatrixCopyConstructor, Matrix)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(mat1);

    EXPECT_TRUE(equal_matricies(mat1, mat2));
}

TEST(MatrixCopyConstructor, Vector)
{
    Vector vec(MAT_1);
    Matrix mat(vec);

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixCopyConstructor, StdVector)
{
    std::vector<real> vec(MAT_1);
    Matrix mat(vec);

    // vec == mat not possible, cannot edit std::vector class
    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixCopyConstructor, InitializerList)
{
    std::vector<real> vec(MAT_1);
    Matrix mat(MAT_1);

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixCopyConstructor, InitializerRowList)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2({
                   {29.9549, -27.7233, 9.8444, 0.0000},
                   {-14.3950, 0.4060, 1.0000, 23.1001},
                   {-43.4291, -61.8100, 46.6723, 99.4774},
                   {-0.2372, 3.5356, 0.0000, 13.7367}
               });

    EXPECT_TRUE(equal_matricies(mat1, mat2));
}

// Assignment operator
TEST(MatrixAssignment, Matrix)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    Matrix mat2;
    mat2 = mat1;

    EXPECT_TRUE(equal_matricies(mat1, mat2));
}

TEST(MatrixAssignment, Vector)
{
    Vector vec(MAT_1);

    Matrix mat;
    mat = vec;

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixAssignment, StdVector)
{
    std::vector<real> vec(MAT_1);

    Matrix mat;
    mat = vec;

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixAssignment, InitializerList)
{
    std::vector<real> vec(MAT_1);
    Matrix mat;
    mat = MAT_1;

    EXPECT_TRUE(equal_matricies(mat, vec));
}

TEST(MatrixAssignment, InitializerRowList)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2;
    mat2 = {
        {29.9549, -27.7233, 9.8444, 0.0000},
        {-14.3950, 0.4060, 1.0000, 23.1001},
        {-43.4291, -61.8100, 46.6723, 99.4774},
        {-0.2372, 3.5356, 0.0000, 13.7367}
    };

    EXPECT_TRUE(equal_matricies(mat1, mat2));
}

// Matrix/Matrix operators
TEST(MatrixMatrixOperators, Equallity)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    ASSERT_TRUE(mat1 == mat1);
    ASSERT_FALSE(mat1 != mat1);
    ASSERT_TRUE(mat1 != mat2);
    ASSERT_FALSE(mat1 == mat2);
}

TEST(MatrixMatrixOperators, Addition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {30.7459, -26.7233, 9.8444, 0.0000,
                               -64.2871, 38.4847, 86.0372, -71.0332,
                               -43.4291, -69.1745, 51.8478, 172.1384,
                               -70.8607, 2.9749, 25.5225, 14.0093});

    EXPECT_TRUE(equal_matricies(expected_mat, (mat1 + mat2)));
}

TEST(MatrixMatrixOperators, AdditionDestAsSrc)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {30.7459, -26.7233, 9.8444, 0.0000,
                               -64.2871, 38.4847, 86.0372, -71.0332,
                               -43.4291, -69.1745, 51.8478, 172.1384,
                               -70.8607, 2.9749, 25.5225, 14.0093});

    mat1 = mat1 + mat2;

    EXPECT_TRUE(equal_matricies(expected_mat, mat1));
}

TEST(MatrixMatrixOperators, Subtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, {29.1639, -28.7233, 9.8444, 0.0000,
                               35.4971, -37.6727, -84.0372, 117.2334,
                               -43.4291, -54.4455, 41.4968, 26.8164,
                               70.3863, 4.0963, -25.5225, 13.4641});

    EXPECT_TRUE(equal_matricies(expected_mat, (mat1 - mat2)));
}

TEST(MatrixMatrixOperators, Multiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, 0.0);

    for (unsigned i = 0; i < MAT_1_ROWS; ++i) {
        for (unsigned j = 0; j < MAT_1_COLS; ++j) {
            for (unsigned k = 0; k < MAT_1_COLS; ++k) {
                expected_mat(i, j) += mat1(i, k) * mat2(k, j);
            }
        }
    }

    EXPECT_TRUE(equal_matricies(expected_mat, (mat1 * mat2)));
}

TEST(MatrixMatrixOperators, MultiplicationDestAsSrc)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix expected_mat(4, 4, 0.0);

    for (unsigned i = 0; i < MAT_1_ROWS; ++i) {
        for (unsigned j = 0; j < MAT_1_COLS; ++j) {
            for (unsigned k = 0; k < MAT_1_COLS; ++k) {
                expected_mat(i, j) += mat1(i, k) * mat2(k, j);
            }
        }
    }

    mat1 = mat1 * mat2;

    EXPECT_TRUE(equal_matricies(expected_mat, mat1));
}

TEST(MatrixMatrixOperators, CumulativeAddition)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 += mat2;

    EXPECT_TRUE(equal_matricies(mat1, (mat3 + mat2)));
}

TEST(MatrixMatrixOperators, CumulativeSubtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 -= mat2;

    EXPECT_TRUE(equal_matricies(mat1, (mat3 - mat2)));
}

TEST(MatrixMatrixOperators, CumulativeMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3 = mat1;
    mat1 *= mat2;

    EXPECT_TRUE(equal_matricies(mat1, (mat3 * mat2)));
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
    Vector res_vec = mat * vec;

    EXPECT_TRUE(equal_matricies(expected_vec, res_vec));
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
//    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
//    std::vector<real> vec(VEC_1);

//    Vector expected_vec({1442.2815, 657.5125, 6090.2054, 230.9662});

//    EXPECT_TRUE(equal_matricies(expected_vec, (mat * vec)));
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

    EXPECT_TRUE(equal_matricies(mat1, mat2 + val));
}

TEST(MatrixScalarOperators, CumulativeSubtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 -= val;

    EXPECT_TRUE(equal_matricies(mat1, mat2 - val));
}

TEST(MatrixScalarOperators, CumulativeMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 *= val;

    EXPECT_TRUE(equal_matricies(mat1, mat2 * val));
}

TEST(MatrixScalarOperators, CumulativeDivision)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    real val = 17.5738;

    Matrix mat2 = mat1;
    mat1 /= val;

    EXPECT_TRUE(equal_matricies(mat1, mat2 / val));
}

// Matrix operators
TEST(MatrixOperators, Transpose)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    Matrix expected_mat(MAT_1_ROWS, MAT_1_COLS, {29.9549, -14.3950, -43.4291, -0.2372,
                                                -27.7233, 0.4060, -61.8100, 3.5356,
                                                9.8444, 1.0000, 46.6723, 0.0000,
                                                0.0000, 23.1001, 99.4774, 13.7367});

    EXPECT_TRUE(equal_matricies(expected_mat, mat.trans()));
}

TEST(MatrixOperators, ElementWiseSubtraction)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3;
    mat3 = mat1 - mat2;

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat3(i), mat1(i) - mat2(i));
    }
}

TEST(MatrixOperators, ElementWiseMultiplication)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3;
    mat3 = Matrix::mulEWise(mat1, mat2);

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat3(i), mat1(i) * mat2(i));
    }
}

TEST(MatrixOperators, ElementWiseMultiplicationOperator)
{
    Matrix mat1(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Matrix mat2(MAT_2_ROWS, MAT_2_COLS, MAT_2);

    Matrix mat3;
    mat3 = mat1 ** mat2;

    for (unsigned i = 0; i < mat1.size(); ++i) {
        EXPECT_FLOAT_EQ(mat3(i), mat1(i) * mat2(i));
    }
}

// Access
TEST(MatrixAccess, Rows)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    EXPECT_EQ(mat.rows(), MAT_1_ROWS);
}

TEST(MatrixAccess, Cols)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    EXPECT_EQ(mat.cols(), MAT_1_COLS);
}

TEST(MatrixAccess, Size)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    EXPECT_EQ(mat.size(), MAT_1_ROWS*MAT_1_COLS);
}

TEST(MatrixAccess, SubscriptParenthesesRowAndCol)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    EXPECT_FLOAT_EQ(mat(0,0), 29.9549);
    EXPECT_FLOAT_EQ(mat(0,3), 0.0000);
    EXPECT_FLOAT_EQ(mat(2,0), -43.4291);
    EXPECT_FLOAT_EQ(mat(1,2), 1.0000);
    EXPECT_FLOAT_EQ(mat(3,3), 13.7367);
}

TEST(MatrixAccess, SubscriptParenthesesIndex)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    EXPECT_FLOAT_EQ(mat(0), 29.9549);
    EXPECT_FLOAT_EQ(mat(3), 0.0000);
    EXPECT_FLOAT_EQ(mat(8), -43.4291);
    EXPECT_FLOAT_EQ(mat(6), 1.0000);
    EXPECT_FLOAT_EQ(mat(15), 13.7367);
}

// Modify
TEST(MatrixModify, AddRow)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);
    mat.addRow(vec);

    Matrix expected_mat({
                            {29.9549, -27.7233, 9.8444, 0.0000},
                            {-14.3950, 0.4060, 1.0000, 23.1001},
                            {-43.4291, -61.8100, 46.6723, 99.4774},
                            {-0.2372, 3.5356, 0.0000, 13.7367},
                            {0.9549, -50.9924, 0.0000, 29.9549}
                        });

    EXPECT_TRUE(equal_matricies(expected_mat, mat));
}

TEST(MatrixModify, AddCol)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);
    Vector vec(VEC_1);
    mat.addCol(vec);

    Matrix expected_mat({
                            {29.9549, -27.7233, 9.8444, 0.0000, 0.9549},
                            {-14.3950, 0.4060, 1.0000, 23.1001, -50.9924},
                            {-43.4291, -61.8100, 46.6723, 99.4774, 0.0000},
                            {-0.2372, 3.5356, 0.0000, 13.7367, 29.9549}
                        });

    EXPECT_TRUE(equal_matricies(expected_mat, mat));
}

// Utility
TEST(MatrixUtility, ToString)
{
    Matrix mat(MAT_1_ROWS, MAT_1_COLS, MAT_1);

    std::string mat_str("[ 29.9549, -27.7233,  9.84440,  0.00000]\n"
                        "[                                      ]\n"
                        "[-14.3950,  0.40600,  1.00000,  23.1001]\n"
                        "[                                      ]\n"
                        "[-43.4291, -61.8100,  46.6723,  99.4774]\n"
                        "[                                      ]\n"
                        "[-0.23720,  3.53560,  0.00000,  13.7367]\n");

    ASSERT_TRUE(mat.str() == mat_str);
}

//// Construct from Matrix
//Matrix mat11(base_mat);
//Matrix mat12 = base_mat;
//Matrix mat13;
//mat13 = base_mat;

//Vector vec11(base_mat);
//Vector vec12 = base_mat;
//Vector vec13;
//vec13 = base_mat;

//// Construct from Vector
//Matrix mat21(base_vec);
//Matrix mat22 = base_vec;
//Matrix mat23;
//mat23 = base_vec;

//Vector vec21(base_vec);
//Vector vec22 = base_vec;
//Vector vec23;
//vec23 = base_vec;

//// Construct from std::vector
//Matrix mat31(base_stdvec);
//Matrix mat32 = base_stdvec;
//Matrix mat33;
//mat33 = base_stdvec;

//Vector vec31(base_stdvec);
//Vector vec32 = base_stdvec;
//Vector vec33;
//vec33 = base_stdvec;

//// Construct from initializer list
//Matrix mat41(VEC_1);
//Matrix mat42 = VEC_1;
//Matrix mat43;
//mat43 = VEC_1;

//Vector vec41(VEC_1);
//Vector vec42 = VEC_1;
//Vector vec43;
//vec43 = VEC_1;

//// Construct from initializer row list
//Matrix mat51(ROW_LIST);
//Matrix mat52 = ROW_LIST;
//Matrix mat53;
//mat53 = ROW_LIST;

////    Vector vec51(ROW_LIST);
////    Vector vec52 = ROW_LIST;
////    Vector vec53;
////    vec53 = ROW_LIST;

//// Function call
//Matrix mat61 = activeFuncMat(base_vec);
//Matrix mat62 = activeFuncMat(base_mat);

//Vector vec61 = activeFuncVec(base_vec);
//Vector vec62 = activeFuncVec(base_mat);


//// Test
//// Always choose Mx1 * 1xN
//// 1xM * Mx1 == a.multiplyElemWise(b).sum()
//Matrix b1 = vec41 * vec41; // Always choose 4x1 * 1x4


//Matrix m1(4, 4);
//Matrix m2(4, 1);
//Matrix m3(1, 4);
//Vector v1(16);
//Vector v2(4);

//m1 + m1; // 4x4  + 4x4      ok
//m1 + m2; // 4x4  + 4x1
//m1 + m3; // 4x4  + 1x4
//m1 + v1; // 4x4  + 16x1     ok, kolla size bara
//m1 + v2; // 4x4  + 4x1
//m2 + m2; // 4x1  + 4x1      ok
//m2 + m3; // 4x1  + 1x4
//m2 + v1; // 4x1  + 16x1
//m2 + v2; // 4x1  + 4x1      ok
//m3 + m3; // 1x4  + 1x4      ok
//m3 + v1; // 1x4  + 16x1
//m3 + v2; // 1x4  + 4x1      ok, kolla size bara
//v1 + v1; // 16x1 + 16x1     ok, kolla size bara
//v1 + v2; // 16x1 + 4x1
//v2 + v2; // 4x1  + 4x1      ok, kolla size bara








