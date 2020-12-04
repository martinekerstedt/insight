#include <Matrix/vectorview.h>
#include <Matrix/vector.h>

VectorView::VectorView(unsigned size) :
    m_src(nullptr),
    m_ptr(nullptr),
    m_size(size)
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
    m_ptr(&(src->vec()[0])),
    m_size(src->rows())
{

}

ExprTrans<VectorView> VectorView::trans() const
{
    return ExprTrans<VectorView>(*const_cast<VectorView*>(this));
}

real VectorView::operator()(const unsigned row, const unsigned col) const
{
    (void)col;

    if (row >= m_size) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= row < " << m_size
                    << "Actual:\n\trow: " << row);
    }

    if (col > 0) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= col < " << 1
                    << "Actual:\n\tcol: " << col);
    }

    return m_ptr[row];
}

real VectorView::operator()(const unsigned idx) const
{
    if (idx >= m_size) {
        THROW_ERROR("Index out of bounds.\n"
                    << "Allowed:\n\t0 <= idx < " << m_size
                    << "Actual:\n\tidx: " << idx);
    }

    return m_ptr[idx];
}

VectorView VectorView::operator==(const Vector *src)
{
    if (m_size != src->rows()) {
        THROW_ERROR("Vector must have same size has VectorView.\n"
                    << m_size
                    << " != "
                    << src->rows());
    }

    m_src = src;
    m_size = src->rows();
    m_ptr = &(src->vec()[0]);

    return *this;
}

//VectorView VectorView::operator==(const VectorView& src)
//{
//    if (this == &src) {
//        return *this;
//    }

//    m_src = src.m_src;
//    m_size = src.m_size;
//    m_ptr = src.m_ptr;

//    return *this;
//}

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
