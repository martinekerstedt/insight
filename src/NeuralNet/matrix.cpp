#include <NeuralNet/matrix.h>
#include <NeuralNet/vector.h>
#include <NeuralNet/vectorview.h>
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
ExprTrans<Matrix> Matrix::trans() const
{
    return ExprTrans<Matrix>(*const_cast<Matrix*>(this));
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

real Matrix::operator()(const unsigned row, const unsigned col) const
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

real Matrix::operator()(const unsigned& idx) const
{
    if (idx >= m_rows*m_cols) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= idx < " << m_rows*m_cols
                    << "Actual:\n\tidx: " << idx);
    }

    return m_vec[idx];
}

VectorView Matrix::row(const unsigned& row) const
{
    VectorView vecView(this, &m_vec[row*m_cols], m_cols);

    return vecView;
}

//Vector Matrix::row(const unsigned& row) const
//{
//    // Could create a vector of references and return it
//    // Then you could access the elements in the matrix
//    // through the vector

//    Vector vec(m_cols);

//    for (unsigned i = 0; i < m_cols; ++i) {
//        vec(i) = m_vec[row*m_cols + i];
//    }

//    return vec;
//}

//Vector Matrix::col(const unsigned& col) const
//{
//    Vector vec(m_rows);

//    for (unsigned i = 0; i < m_rows; ++i) {
//        vec(i) = m_vec[i*m_cols + col];
//    }

//    return vec;
//}

void Matrix::fill(real val)
{
    std::fill(m_vec.begin(), m_vec.end(), val);
}

void Matrix::resize(unsigned rows, unsigned cols, real val)
{
    m_vec.resize(rows*cols, val);
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

        if (num < -9999.99) {
            ss << std::setprecision(1) << num;
        } else if (num < -999.999) {
            ss << std::setprecision(2) << num;
        } else if (num < -99.9999) {
            ss << std::setprecision(3) << num;
        } else if (num < -9.99999) {
            ss << std::setprecision(4) << num;
        } else {
            ss << std::setprecision(5) << num;
        }

    } else {

        if (num > 9999.99) {
            ss << " " << std::setprecision(1) << num;
        } else if (num > 999.999) {
            ss << " " << std::setprecision(2) << num;
        } else if (num > 99.9999) {
            ss << " " << std::setprecision(3) << num;
        } else if (num > 9.99999) {
            ss << " " << std::setprecision(4) << num;
        } else {
            ss << " " << std::setprecision(5) << num;
        }

    }

    return ss.str();
}

std::string Matrix::str()
{
    // Get maximum number of digts
    // Set precision to 10 - max digits

    std::stringstream ss;

    for (unsigned i = 0; i < m_rows; ++i) {

        ss << "[" << num2str(m_vec[i*m_cols]);

        for (unsigned j = 1; j < m_cols; ++j) {

            ss << ", " << num2str(m_vec[i*m_cols + j]);
        }


        if (i < (m_rows - 1)) {
            ss << "]\n[";

            for (unsigned j = 0; j < ((m_cols*10) - 2); ++j) {
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


















