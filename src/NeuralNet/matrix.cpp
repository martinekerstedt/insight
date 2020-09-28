#include <NeuralNet/matrix.h>
#include <NeuralNet/vector.h>
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
    m_vec.resize(m_rows*m_cols);
}

Matrix::Matrix(unsigned rows, unsigned cols, const real& initVal) :
    m_rows(rows),
    m_cols(cols)
{
    m_vec.resize(m_rows*m_cols, initVal);
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

Matrix::Matrix(unsigned rows, unsigned cols, const std::initializer_list<real> &list) :
    m_rows(rows),
    m_cols(cols),
    m_vec(list)
{

}

Matrix::Matrix(const std::vector<real>& vec) :
    m_rows(vec.size()),
    m_cols(1),
    m_vec(vec)
{

}

Matrix::Matrix(const std::initializer_list<real>& list) :
    m_rows(list.size()),
    m_cols(1),
    m_vec(list)
{

}

Matrix::Matrix(const std::initializer_list<std::initializer_list<real>>& row_list) :
    m_rows(row_list.size()),
    m_cols(row_list.begin()->size())
{
    m_vec.resize(m_rows*m_cols);

    unsigned i = 0;
    unsigned j = 0;

    for (const auto& row : row_list) {

        j = 0;

        for (const auto& elem : row) {
            (*this)(i, j) = elem;
            ++j;
        }

        ++i;
    }
}

// Matrix/Matrix operations
bool Matrix::operator==(const Matrix &rhs) const
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

bool Matrix::operator!=(const Matrix &rhs) const
{
    return !(*this == rhs);
}

//Matrix Matrix::operator+(const Matrix& rhs) const
//{
//    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
//        THROW_ERROR("Matrix sizes must be equal. "
//                    << m_rows << "x" << m_cols
//                    << " != "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] + rhs.m_vec[i];
//    }

//    return res;
//}

//Matrix Matrix::operator-(const Matrix& rhs) const
//{
//    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
//        THROW_ERROR("Matrix sizes must be equal. "
//                    << m_rows << "x" << m_cols
//                    << " != "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] - rhs.m_vec[i];
//    }

//    return res;
//}

//Matrix Matrix::operator*(const Matrix& rhs) const
//{
//    if (m_cols != rhs.m_rows) {
//        THROW_ERROR("Matrices have incompatible dimensions. "
//                    << this->m_rows << "x" << this->m_cols
//                    << " and "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

//    Matrix res(m_rows, rhs.m_cols, 0.0);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < rhs.m_cols; ++j) {
//            for (unsigned k = 0; k < rhs.m_rows; ++k) {
//                res(i, j) += (*this)(i, k) * rhs(k, j);
//            }
//        }
//    }


//    return res;
//}

//Matrix& Matrix::operator+=(const Matrix& rhs)
//{
//    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
//        THROW_ERROR("Matrix sizes must be equal. "
//                    << m_rows << "x" << m_cols
//                    << " != "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        m_vec[i] += rhs.m_vec[i];
//    }

//    return *this;
//}

//Matrix& Matrix::operator-=(const Matrix& rhs)
//{
//    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
//        THROW_ERROR("Matrix sizes must be equal. "
//                    << m_rows << "x" << m_cols
//                    << " != "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        m_vec[i] -= rhs.m_vec[i];
//    }

//    return *this;
//}

//Matrix& Matrix::operator*=(const Matrix& rhs)
//{
//    Matrix res = (*this) * rhs;
//    (*this) = res;
//    return *this;
//}

//// Matrix/scalar operations
//Matrix Matrix::operator+(const real& rhs) const
//{
//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] + rhs;
//    }

//    return res;
//}

//Matrix Matrix::operator-(const real& rhs) const
//{
//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] - rhs;
//    }

//    return res;
//}

//Matrix Matrix::operator*(const real& rhs) const
//{
//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] * rhs;
//    }

//    return res;
//}

//Matrix Matrix::operator/(const real& rhs) const
//{
//    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        res.m_vec[i] = m_vec[i] / rhs;
//    }

//    return res;
//}

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
MatExprTrans<Matrix> Matrix::trans() const
{
    return MatExprTrans<Matrix>(*const_cast<Matrix*>(this));
}


//Matrix Matrix::transpose() const
//{
//    Matrix res(m_cols, m_rows);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        for (unsigned j = 0; j < m_cols; ++j) {
//            res(j, i) = m_vec[i*m_cols + j];
//        }
//    }

//    return res;
//}

//Matrix& Matrix::multiplyElemWise(const Matrix &rhs)
//{
//    if ((m_rows != rhs.m_rows) || (m_cols != rhs.m_cols)) {
//        THROW_ERROR("Matrix sizes must be equal. "
//                    << m_rows << "x" << m_cols
//                    << " != "
//                    << rhs.m_rows << "x" << rhs.m_cols
//                    << ".");
//    }

////    Matrix res(m_rows, m_cols);

//    for (unsigned i = 0; i < (m_rows*m_cols); ++i) {
//        m_vec[i] = m_vec[i] * rhs.m_vec[i];
//    }

//    return *this;
//}

// Access
//real* Matrix::data()
//{
//    return m_vec.data();
//}

//const real* Matrix::data() const
//{
//    return m_vec.data();
//}

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
    if (row >= m_rows) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= row < " << m_rows
                    << "Actual:\n\trow: " << row);
    }

    if (col >= m_cols) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= col < " << m_cols
                    << "Actual:\n\tcol: " << col);
    }

    return m_vec[row*m_cols + col];
}

real& Matrix::operator()(const unsigned& idx)
{
    if (idx >= m_rows*m_cols) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= idx < " << m_rows*m_cols
                    << "Actual:\n\tidx: " << idx);
    }

    return m_vec[idx];
}

const real& Matrix::operator()(const unsigned& row, const unsigned& col) const
{
    if (row >= m_rows) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= row < " << m_rows
                    << "Actual:\n\trow: " << row);
    }

    if (col >= m_cols) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= col < " << m_cols
                    << "Actual:\n\tcol: " << col);
    }

    return m_vec[row*m_cols + col];
}

const real& Matrix::operator()(const unsigned& idx) const
{
    if (idx >= m_rows*m_cols) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= idx < " << m_rows*m_cols
                    << "Actual:\n\tidx: " << idx);
    }

    return m_vec[idx];
}

Vector Matrix::row(const unsigned& row) const
{
    // Could create a vector of references and return it
    // Then you could access the elements in the matrix
    // through the vector

    Vector vec(m_cols);

    for (unsigned i = 0; i < m_cols; ++i) {
        vec(i) = m_vec[row*m_cols + i];
    }

    return vec;
}

Vector Matrix::col(const unsigned& col) const
{
    Vector vec(m_rows);

    for (unsigned i = 0; i < m_rows; ++i) {
        vec(i) = m_vec[i*m_cols + col];
    }

    return vec;
}

void Matrix::resize(unsigned rows, unsigned cols)
{
    m_vec.resize(rows*cols);
    m_rows = rows;
    m_cols = cols;
}

void Matrix::reserve(unsigned size)
{
    m_vec.reserve(size);
}

// Modifiers
void Matrix::addRow(const Vector& row)
{
    if (m_cols != row.size()) {
        THROW_ERROR("Number of matrix cols must equal Vector size. "
                    << m_cols << " != " << row.size() << ".");
    }

    for (unsigned i = 0; i < m_cols; ++i) {
        m_vec.push_back(row(i));
    }

    ++m_rows;
}

void Matrix::addCol(const Vector& col)
{
    if (m_rows != col.size()) {
        THROW_ERROR("Number of matrix rows must equal Vector size. "
                    << m_rows << " != " << col.size() << ".");
    }

    // Single row case
    if (m_rows == 1) {
        m_vec.push_back(col(0));
        ++m_cols;
        return;
    }

    // Allocate
    for (unsigned i = 0; i < m_rows; ++i) {
        m_vec.push_back(0);
    }

    // Increment number of cols and shift elements to correct place
    ++m_cols;

    for (int i = (m_rows - 1); i >= 0; --i) {

        m_vec[i*m_cols + (m_cols - 1)] = col(i);

        for (int j = (m_cols - 2); j >= 0; --j) {

            m_vec[i*m_cols + j] = m_vec[i*m_cols + j - i];
        }
    }
}

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

bool Matrix::sourceOk(const Matrix &destMat)
{
    return !(&destMat == this);
}


















