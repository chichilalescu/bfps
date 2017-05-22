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



#include <vector>
#include "base.hpp"
#include "field_layout.hpp"

#ifndef FIELD_BINARY_IO_HPP

#define FIELD_BINARY_IO_HPP

template <MPI_Datatype element_type, field_components fc>
class field_binary_IO:public field_layout<fc>
{
    public:
        MPI_Datatype mpi_array_dtype;

        /* methods */
        field_binary_IO(
                const hsize_t *SIZES,
                const hsize_t *SUBSIZES,
                const hsize_t *STARTS,
                const MPI_Comm COMM_TO_USE);
        ~field_binary_IO();
};

#endif//FIELD_BINARY_IO_HPP

