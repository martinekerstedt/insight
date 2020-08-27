#include <NeuralNet/vector.h>
#include <Common/types.h>

#include <NeuralNet/matrix.h>

// Constructors
Vector::Vector() :
    m_size(0)
{

}

Vector::Vector(unsigned size) :
    m_size(size)
{
    m_vec.resize(size);
}

Vector::Vector(unsigned size, const real& initVal) :
    m_size(size)
{
    m_vec.resize(size, initVal);
}

// Copy Constructors
Vector::Vector(const Vector& rhs) :
    m_size(rhs.m_size),
    m_vec(rhs.m_vec)
{

}

Vector::Vector(const Matrix& rhs) :
    m_size(rhs.size()),
    m_vec(rhs.vec())
{

}

Vector::Vector(const std::vector<real>& rhs) :
    m_size(rhs.size()),
    m_vec(rhs)
{

}

// Destructor
Vector::~Vector()
{

}

// Assignment operators
Vector& Vector::operator=(const Vector& rhs)
{
    if (&rhs == this) {
        return *this;
    }

    m_size = rhs.m_size;
    m_vec.resize(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] = rhs.m_vec[i];
    }

    return *this;
}

Vector& Vector::operator=(const Matrix& rhs)
{
    m_size = rhs.size();
    m_vec.resize(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] = rhs.vec()[i];
    }

    return *this;
}

Vector& Vector::operator=(const std::vector<real>& rhs)
{
    m_size = rhs.size();
    m_vec.resize(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] = rhs[i];
    }

    return *this;
}

// Vector/Vector operations
bool Vector::operator==(const Vector &rhs)
{
    if (m_size != rhs.m_size) {
        return false;
    }

    for (unsigned i = 0; i < m_vec.size(); ++i) {
        if (std::abs(m_vec[i] - rhs.m_vec[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

bool Vector::operator!=(const Vector &rhs)
{
    return !(*this == rhs);
}

Vector Vector::operator+(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] + rhs.m_vec[i];
    }

    return res;
}

Vector Vector::operator-(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] - rhs.m_vec[i];
    }

    return res;
}

Vector Vector::operator*(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] * rhs.m_vec[i];
    }

    return res;
}

Vector Vector::operator/(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] / rhs.m_vec[i];
    }

    return res;
}

Vector& Vector::operator+=(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] += rhs.m_vec[i];
    }

    return *this;
}

Vector& Vector::operator-=(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] -= rhs.m_vec[i];
    }

    return *this;
}

Vector& Vector::operator*=(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] *= rhs.m_vec[i];
    }

    return *this;
}

Vector& Vector::operator/=(const Vector& rhs)
{
    if (m_size != rhs.m_size) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.m_size << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] /= rhs.m_vec[i];
    }

    return *this;
}

Matrix Vector::matMul(const Vector &rhs)
{
    Matrix mat(m_size, rhs.m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        for (unsigned j = 0; j < rhs.m_size; ++j) {
            mat(i, j) = m_vec[i] * rhs.m_vec[j];
        }
    }

    return mat;
}

// Vector/std::vector operations
bool Vector::operator==(const std::vector<real> &rhs)
{
    if (m_size != rhs.size()) {
        return false;
    }

    for (unsigned i = 0; i < m_vec.size(); ++i) {
        if (std::abs(m_vec[i] - rhs[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

bool Vector::operator!=(const std::vector<real> &rhs)
{
    return !(*this == rhs);
}

Vector Vector::operator+(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] + rhs[i];
    }

    return res;
}

Vector Vector::operator-(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] - rhs[i];
    }

    return res;
}

Vector Vector::operator*(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] * rhs[i];
    }

    return res;
}

Vector Vector::operator/(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] / rhs[i];
    }

    return res;
}

Vector& Vector::operator+=(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] += rhs[i];
    }

    return *this;
}

Vector& Vector::operator-=(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] -= rhs[i];
    }

    return *this;
}

Vector& Vector::operator*=(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] *= rhs[i];
    }

    return *this;
}

Vector& Vector::operator/=(const std::vector<real>& rhs)
{
    if (m_size != rhs.size()) {
        THROW_ERROR("Vector sizes must be equal. "
                    << m_size
                    << " != "
                    << rhs.size() << ".");
    }

    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] /= rhs[i];
    }

    return *this;
}

// Vector/scalar operations
Vector Vector::operator+(const real& rhs)
{
    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] + rhs;
    }

    return res;
}

Vector Vector::operator-(const real& rhs)
{
    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] - rhs;
    }

    return res;
}

Vector Vector::operator*(const real& rhs)
{
    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] * rhs;
    }

    return res;
}

Vector Vector::operator/(const real& rhs)
{
    Vector res(m_size);

    for (unsigned i = 0; i < m_size; ++i) {
        res.m_vec[i] = m_vec[i] / rhs;
    }

    return res;
}

Vector& Vector::operator+=(const real& rhs)
{
    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] += rhs;
    }

    return *this;
}

Vector& Vector::operator-=(const real& rhs)
{
    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] -= rhs;
    }

    return *this;
}

Vector& Vector::operator*=(const real& rhs)
{
    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] *= rhs;
    }

    return *this;
}

Vector& Vector::operator/=(const real& rhs)
{
    for (unsigned i = 0; i < m_size; ++i) {
        m_vec[i] /= rhs;
    }

    return *this;
}

// Vector/Matrix operations
bool Vector::operator==(const Matrix &rhs)
{
    if (m_size != rhs.rows()) {

        if (m_size != rhs.cols()) {
            return false;
        } else if (rhs.rows() != 1) {
            return false;
        }

    } else if (rhs.cols() != 1) {
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

bool Vector::operator!=(const Matrix &rhs)
{
    return !(*this == rhs);
}

Vector Vector::operator*(const Matrix& rhs)
{
    if (m_size != rhs.rows()) {
        THROW_ERROR("Vector and Matrix have incompatible dimensions. "
                    << m_size
                    << " != "
                    << rhs.rows() << ".");
    }

    Vector res(rhs.cols(), 0.0);

    for (unsigned i = 0; i < rhs.rows(); ++i) {
        for (unsigned j = 0; j < rhs.cols(); ++j) {
            res.vec()[j] += m_vec[i] * rhs.vec()[i*rhs.cols() + j];
        }
    }

    return res;
}

Vector& Vector::operator*=(const Matrix& rhs)
{
    if (m_size != rhs.rows()) {
        THROW_ERROR("Vector and Matrix have incompatible dimensions. "
                    << m_size
                    << " != "
                    << rhs.rows() << ".");
    }

    Vector res(rhs.cols(), 0.0);

    for (unsigned i = 0; i < rhs.rows(); ++i) {
        for (unsigned j = 0; j < rhs.cols(); ++j) {
            res.vec()[j] += m_vec[i] * rhs.vec()[i*rhs.cols() + j];
        }
    }

    *this = res;
    return *this;
}

// Access
std::vector<real>& Vector::vec()
{
    return m_vec;
}

const std::vector<real>& Vector::vec() const
{
    return m_vec;
}

real& Vector::operator()(const unsigned& idx)
{
    return m_vec[idx];
}

const real& Vector::operator()(const unsigned& idx) const
{
    return m_vec[idx];
}

unsigned Vector::size() const
{
    return m_vec.size();
}

real& Vector::front()
{
    return m_vec.front();
}

real& Vector::back()
{
    return m_vec.back();
}

void Vector::pushBack(const real& val)
{
    m_vec.push_back(val);
    ++m_size;
}

void Vector::pushBack(real&& val)
{
    m_vec.push_back(val);
    ++m_size;
}

real Vector::popBack()
{
    real res = m_vec.back();
    --m_size;
    m_vec.pop_back();
    return res;
}

void Vector::clear() noexcept
{
    m_size = 0;
    m_vec.clear();
}


























































