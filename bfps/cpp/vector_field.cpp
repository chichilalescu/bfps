#include "vector_field.hpp"

/* destructor doesn't actually do anything */
template <class rnumber>
vector_field<rnumber>::~vector_field()
{}


template <class rnumber>
vector_field<rnumber>::vector_field(
        field_descriptor<rnumber> *d,
        rnumber *data)
{
    this->is_real = true;
    this->cdata = (rnumber (*)[2])(data);
    this->rdata = data;
    this->descriptor = d;
}

template <class rnumber>
vector_field<rnumber>::vector_field(
        field_descriptor<rnumber> *d,
        rnumber (*data)[2])
{
    this->is_real = false;
    this->rdata = (rnumber*)(&data[0][0]);
    this->cdata = data;
    this->descriptor = d;
}

template <class rnumber>
vector_field<rnumber>& vector_field<rnumber>::operator*(rnumber factor)
{
    ptrdiff_t i;
    for (i = 0;
         i < this->descriptor->local_size * 2;
         i++)
        *(this->rdata + i) *= factor;
    return *this;
}

template class vector_field<float>;
template class vector_field<double>;
