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

#include "field_descriptor.hpp"

template <class rnumber>
class vector_field
{
    private:
        field_descriptor<rnumber> *descriptor;
        rnumber *rdata;
        rnumber (*cdata)[2];
        bool is_real;
    public:
        vector_field(field_descriptor<rnumber> *d, rnumber *data);
        vector_field(field_descriptor<rnumber> *d, rnumber (*data)[2]);
        ~vector_field();

        /* various operators */
        vector_field &operator*(rnumber factor);
};

