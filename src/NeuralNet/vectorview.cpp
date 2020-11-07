#include <NeuralNet/vectorview.h>
#include <NeuralNet/vector.h>

VectorView::VectorView() :
    m_src(nullptr),
    m_ptr(nullptr),
    m_size(0)
{

}

VectorView::VectorView(const Matrix* src, const real* ptr, unsigned size) :
    m_src(src),
    m_ptr(ptr),
    m_size(size)
{

}

VectorView::VectorView(const Vector *src) :
    m_src(src),
    m_ptr(&src->vec()[0]),
    m_size(src->size())
{

}

MatExprTrans<VectorView> VectorView::trans() const
{
    return MatExprTrans<VectorView>(*const_cast<VectorView*>(this));
}

const real& VectorView::operator()(const unsigned& row, const unsigned& col) const
{
    if (row >= m_size) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= row < " << m_size
                    << "Actual:\n\trow: " << row);
    }

    if (col >= 1) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= col < " << 1
                    << "Actual:\n\tcol: " << col);
    }

    return m_ptr[row];
}

const real& VectorView::operator()(const unsigned& idx) const
{
    if (idx >= m_size) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= idx < " << m_size
                    << "Actual:\n\tidx: " << idx);
    }

    return m_ptr[idx];
}

unsigned VectorView::rows() const
{
    return m_size;
}

unsigned VectorView::cols() const
{
    return 1;
}

unsigned VectorView::evalCost() const
{
    return 1;
}

unsigned VectorView::size() const
{
    return m_size;
}

bool VectorView::sourceOk(const Matrix &destMat)
{
    return !(&destMat == m_src);
}

void VectorView::cache()
{
    return;
}
