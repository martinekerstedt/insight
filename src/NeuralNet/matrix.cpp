#include <NeuralNet/matrix.h>
#include <Common/types.h>
#include <sstream>
#include <iomanip>

// Constructors
Matrix::Matrix() :
    m_rows(0),
    m_cols(0)
{

}

Matrix::Matrix(unsigned rows, unsigned cols) :
    m_rows(rows),
    m_cols(cols)
{
    m_vec.resize(rows*cols);
}

Matrix::Matrix(unsigned rows, unsigned cols, const real& initVal) :
    m_rows(rows),
    m_cols(cols)
{
    m_vec.resize(rows*cols, initVal);
}

Matrix::Matrix(unsigned rows, unsigned cols, const Vector& initVals) :
    m_rows(rows),
    m_cols(cols),
    m_vec(initVals.vec())
{

}

Matrix::Matrix(unsigned rows, unsigned cols, const std::vector<real>& initVals) :
    m_rows(rows),
    m_cols(cols),
    m_vec(initVals)
{

}

// Copy Constructor
Matrix::Matrix(const Matrix& mat) :
    m_rows(mat.m_rows),
    m_cols(mat.m_cols),
    m_vec(mat.m_vec)
{

}

Matrix::Matrix(const Vector& vec) :
    m_rows(vec.size()),
    m_cols(1),
    m_vec(vec.vec())
{

}

Matrix::Matrix(const std::vector<real>& vec) :
    m_rows(vec.size()),
    m_cols(1),
    m_vec(vec)
{

}

// Destructor
Matrix::~Matrix()
{

}

// Assignment operators
Matrix& Matrix::operator=(const Matrix& rhs)
{
    if (&rhs == this) {
        return *this;
    }

    m_rows = rhs.m_rows;
    m_cols = rhs.m_cols;
    m_vec.resize(m_rows*m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] = rhs.m_vec[i];
    }

    return *this;
}

Matrix& Matrix::operator=(const Vector& rhs)
{
    m_rows = rhs.size();
    m_cols = 1;
    m_vec.resize(m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        m_vec[i] = rhs.vec()[i];
    }

    return *this;
}

Matrix& Matrix::operator=(const std::vector<real>& rhs)
{
    m_rows = rhs.size();
    m_cols = 1;
    m_vec.resize(m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        m_vec[i] = rhs[i];
    }

    return *this;
}

// Matrix/Matrix operations
bool Matrix::operator==(const Matrix &rhs)
{
    if (m_rows != rhs.m_rows) {
        return false;
    }

    if (m_cols != rhs.m_cols) {
        return false;
    }

    if (m_vec.size() != rhs.m_vec.size()) {
        return false;
    }

    for (unsigned i = 0; i < m_vec.size(); ++i) {
        if (std::abs(m_vec[i] - rhs.m_vec[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

bool Matrix::operator!=(const Matrix &rhs)
{
    return !(*this == rhs);
}

Matrix Matrix::operator+(const Matrix& rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] + rhs.m_vec[i];
    }

    return res;
}

Matrix Matrix::operator-(const Matrix& rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] - rhs.m_vec[i];
    }

    return res;
}

Matrix Matrix::operator*(const Matrix& rhs)
{
    if (m_cols != rhs.m_rows) {
        THROW_ERROR("Matrices have incompatible dimensions. "
                    << this->m_rows << "x" << this->m_cols
                    << " and "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    Matrix res(m_rows, rhs.m_cols, 0.0);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < rhs.m_cols; ++j) {
            for (unsigned k = 0; k < m_rows; ++k) {
                res.m_vec[i*rhs.m_cols + j] += m_vec[i*m_cols + k] * rhs.m_vec[k*rhs.m_cols + j];
            }
        }
    }

    return res;
}

Matrix& Matrix::operator+=(const Matrix& rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] += rhs.m_vec[i];
    }

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] -= rhs.m_vec[i];
    }

    return *this;
}

Matrix& Matrix::operator*=(const Matrix& rhs)
{
    Matrix res = (*this) * rhs;
    (*this) = res;
    return *this;
}


// Matrix/Vector operations
bool Matrix::operator==(const Vector &rhs)
{
    if (m_rows != rhs.size()) {

        if (m_cols != rhs.size()) {
            return false;
        } else if (m_rows != 1) {
            return false;
        }

    } else if (m_cols != 1) {
        return false;
    }

    if (m_vec.size() != rhs.vec().size()) {
        return false;
    }

    for (unsigned i = 0; i < m_vec.size(); ++i) {
        if (std::abs(m_vec[i] - rhs.vec()[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

bool Matrix::operator!=(const Vector &rhs)
{
    return !(*this == rhs);
}

Vector Matrix::operator*(const Vector& rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Vector res(m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.vec()[i] += m_vec[i*m_cols + j] * rhs.vec()[j];
        }
    }

    return res;
}

Matrix Matrix::addRowWise(const Vector &rhs)
{
    if (m_rows != rhs.size()) {
        THROW_ERROR("Number of matrix rows must equal Vector size. "
                    << m_rows << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] + rhs.vec()[i];
        }
    }

    return res;
}

Matrix Matrix::addColWise(const Vector &rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] + rhs.vec()[j];
        }
    }

    return res;
}

Matrix Matrix::subtractRowWise(const Vector &rhs)
{
    if (m_rows != rhs.size()) {
        THROW_ERROR("Number of matrix rows must equal Vector size. "
                    << m_rows << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] - rhs.vec()[i];
        }
    }

    return res;
}

Matrix Matrix::subtractColWise(const Vector &rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] - rhs.vec()[j];
        }
    }

    return res;
}

Matrix Matrix::multiplyRowWise(const Vector &rhs)
{
    if (m_rows != rhs.size()) {
        THROW_ERROR("Number of matrix rows must equal Vector size. "
                    << m_rows << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] * rhs.vec()[i];
        }
    }

    return res;
}

Matrix Matrix::multiplyColWise(const Vector &rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] * rhs.vec()[j];
        }
    }

    return res;
}

Matrix Matrix::divideRowWise(const Vector &rhs)
{
    if (m_rows != rhs.size()) {
        THROW_ERROR("Number of matrix rows must equal Vector size. "
                    << m_rows << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] / rhs.vec()[i];
        }
    }

    return res;
}

Matrix Matrix::divideColWise(const Vector &rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.m_vec[i*m_cols + j] = m_vec[i*m_cols + j] / rhs.vec()[j];
        }
    }

    return res;
}

// Matrix/std::vector operations
bool Matrix::operator==(const std::vector<real> &rhs)
{
    if (m_rows != rhs.size()) {

        if (m_cols != rhs.size()) {
            return false;
        } else if (m_rows != 1) {
            return false;
        }

    } else if (m_cols != 1) {
        return false;
    }

    if (m_vec.size() != rhs.size()) {
        return false;
    }

    for (unsigned i = 0; i < m_vec.size(); ++i) {
        if (std::abs(m_vec[i] - rhs[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

bool Matrix::operator!=(const std::vector<real> &rhs)
{
    return !(*this == rhs);
}

Vector Matrix::operator*(const std::vector<real>& rhs)
{
    if (m_cols != rhs.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << rhs.size() << ".");
    }

    Vector res(m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res.vec()[i] += m_vec[i*m_cols + j] * rhs[j];
        }
    }

    return res;
}

// Matrix/scalar operations
Matrix Matrix::operator+(const real& rhs)
{
    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] + rhs;
    }

    return res;
}

Matrix Matrix::operator-(const real& rhs)
{
    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] - rhs;
    }

    return res;
}

Matrix Matrix::operator*(const real& rhs)
{
    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] * rhs;
    }

    return res;
}

Matrix Matrix::operator/(const real& rhs)
{
    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] / rhs;
    }

    return res;
}

Matrix& Matrix::operator+=(const real& rhs)
{
    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] += rhs;
    }

    return *this;
}

Matrix& Matrix::operator-=(const real& rhs)
{
    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] -= rhs;
    }

    return *this;
}

Matrix& Matrix::operator*=(const real& rhs)
{
    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] *= rhs;
    }

    return *this;
}

Matrix& Matrix::operator/=(const real& rhs)
{
    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        m_vec[i] /= rhs;
    }

    return *this;
}

// Matrix operations
Matrix Matrix::transpose()
{
    Matrix res(m_cols, m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        for (unsigned j = 0; j < m_cols; ++j) {
            res(j, i) = m_vec[i*m_cols + j];
        }
    }

    return res;
}

Matrix Matrix::subtractElemWise(const Matrix &rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] - rhs.m_vec[i];
    }

    return res;
}

Matrix Matrix::multiplyElemWise(const Matrix &rhs)
{
    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
        THROW_ERROR("Matrix sizes must be equal. "
                    << m_rows << "x" << m_cols
                    << " != "
                    << rhs.m_rows << "x" << rhs.m_cols
                    << ".");
    }

    Matrix res(m_rows, m_cols);

    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
        res.m_vec[i] = m_vec[i] * rhs.m_vec[i];
    }

    return res;
}

// Access
std::vector<real>& Matrix::vec()
{
    return m_vec;
}

const std::vector<real>& Matrix::vec() const
{
    return m_vec;
}

unsigned Matrix::rows() const
{
    return m_rows;
}

unsigned Matrix::cols() const
{
    return m_cols;
}

unsigned Matrix::size() const
{
    return m_rows*m_cols;
}

real& Matrix::operator()(const unsigned& row, const unsigned& col)
{
    return this->m_vec[row*m_cols + col];
}

real& Matrix::operator()(const unsigned& idx)
{
    return this->m_vec[idx];
}

const real& Matrix::operator()(const unsigned& row, const unsigned& col) const
{
    return this->m_vec[row*m_cols + col];
}

const real& Matrix::operator()(const unsigned& idx) const
{
    return this->m_vec[idx];
}

// Modifiers


std::string Matrix::num2str(real num)
{
    std::stringstream ss;
    ss << std::fixed;

    if (num < 0) {

        if (num < -99.99) {
            ss << std::setprecision(1) << num;
        } else if (num < -9.999) {
            ss << std::setprecision(2) << num;
        } else {
            ss << std::setprecision(3) << num;
        }

    } else {

        if (num > 99.99) {
            ss << " " << std::setprecision(1) << num;
        } else if (num > 9.999) {
            ss << " " << std::setprecision(2) << num;
        } else {
            ss << " " << std::setprecision(3) << num;
        }

    }

    return ss.str();
}

std::string Matrix::str()
{
    std::stringstream ss;
//    ss << std::fixed << std::setprecision(2);

    for (unsigned i = 0; i < m_rows; ++i) {

        ss << "[" << num2str(m_vec[i*m_cols]);

        for (unsigned j = 1; j < m_cols; ++j) {

            ss << ", " << num2str(m_vec[i*m_cols + j]);
        }


        if (i < (m_rows - 1)) {
            ss << "]\n[";

            for (unsigned j = 0; j < ((m_cols*8) - 2); ++j) {
                ss << " ";
            }

            ss << "]\n";
        } else {
            ss << "]\n";
        }

    }

    return ss.str();
}






























































//Matrix Matrix::operator+(const Vector& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            unsigned idx = i*m_cols + j;
//            res.m_vec[idx] = m_vec[idx] + rhs.vec()[i];
//        }
//    }

//    return res;
//}

//Matrix Matrix::operator-(const Vector& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            unsigned idx = i*m_cols + j;
//            res.m_vec[idx] = m_vec[idx] - rhs.vec()[i];
//        }
//    }

//    return res;
//}

//Matrix Matrix::operator*(const Vector& rhs)
//{
//    if (m_cols != rhs.size()) {
//        THROW_ERROR("Matrices have incompatible dimensions. "
//                    << m_rows << "x" << m_cols
//                    << " and "
//                    << rhs.size() << "x1"
//                    << ".");
//    }

//    Matrix res(m_rows, 1);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            res.m_vec[i] += m_vec[i*m_cols + j] * rhs.vec()[j];
//        }
//    }

//    return res;
//}

//Matrix& Matrix::operator+=(const Vector& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            m_vec[i*m_cols + j] += rhs.vec()[i];
//        }
//    }

//    return *this;
//}

//Matrix& Matrix::operator-=(const Vector& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            m_vec[i*m_cols + j] -= rhs.vec()[i];
//        }
//    }

//    return *this;
//}

//Matrix& Matrix::operator*=(const Vector& rhs)
//{
//    Matrix res = (*this) * rhs;
//    (*this) = res;
//    return *this;
//}

// Matrix/std::vector operations
//Matrix Matrix::operator+(const std::vector<real>& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            unsigned idx = i*m_cols + j;
//            res.m_vec[idx] = m_vec[idx] + rhs[i];
//        }
//    }

//    return res;
//}

//Matrix Matrix::operator-(const std::vector<real>& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            unsigned idx = i*m_cols + j;
//            res.m_vec[idx] = m_vec[idx] - rhs[i];
//        }
//    }

//    return res;
//}

//Matrix Matrix::operator*(const std::vector<real>& rhs)
//{
//    if (m_cols != rhs.size()) {
//        THROW_ERROR("Matrices have incompatible dimensions. "
//                    << m_rows << "x" << m_cols
//                    << " and "
//                    << rhs.size() << "x1"
//                    << ".");
//    }

//    Matrix res(m_rows, 1);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            res.m_vec[i*m_cols + j] += m_vec[i*m_cols + j] * rhs[j];
//        }
//    }

//    return res;
//}

//Matrix& Matrix::operator+=(const std::vector<real>& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            m_vec[i*m_cols + j] += rhs[i];
//        }
//    }

//    return *this;
//}

//Matrix& Matrix::operator-=(const std::vector<real>& rhs)
//{
//    if (m_rows != rhs.size()) {
//        THROW_ERROR("Matrix rows must be equal Vector rows. "
//                    << m_rows << " != "
//                    << rhs.size() << ".");
//    }

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            m_vec[i*m_cols + j] -= rhs[i];
//        }
//    }

//    return *this;
//}

//Matrix& Matrix::operator*=(const std::vector<real>& rhs)
//{
//    Matrix res = (*this) * rhs;
//    (*this) = res;
//    return *this;
//}
