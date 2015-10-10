/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/

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
