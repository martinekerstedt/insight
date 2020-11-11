#ifndef VECTORVIEW_H
#define VECTORVIEW_H

#include <Matrix/expr_base.h>

class Vector;

class VectorView final : private ExprInterface
{
public:
    VectorView();

    VectorView(const Matrix* src,
               const real* ptr,
               unsigned size);

    VectorView(const Vector* src);

    ExprTrans<VectorView> trans() const;

    real operator()(const unsigned row, const unsigned col) const;
    real operator()(const unsigned idx) const;

    unsigned rows() const;
    unsigned cols() const;
    unsigned evalCost() const;
    unsigned size() const;
    bool sourceOk(const Matrix &destMat);
    void cache();

private:
    const Matrix* m_src;
    const real* m_ptr;
    unsigned m_size;

};

#endif // VECTORVIEW_H
