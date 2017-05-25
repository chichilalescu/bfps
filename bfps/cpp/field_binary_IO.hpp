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
#include <string>
#include "base.hpp"
#include "fftw_interface.hpp"
#include "field_layout.hpp"
#include "field.hpp"

#ifndef FIELD_BINARY_IO_HPP

#define FIELD_BINARY_IO_HPP

/* could this be a boolean somehow?*/
enum field_representation: bool {
    REAL = true,
    COMPLEX = false};

template <typename rnumber>
constexpr MPI_Datatype mpi_type(
        field_representation fr)
{
    return ((fr == REAL) ?
            mpi_real_type<rnumber>::real() :
            mpi_real_type<rnumber>::complex());
}

template <typename rnumber, field_representation fr, field_components fc>
class field_binary_IO:public field_layout<fc>
{
    private:
        MPI_Comm io_comm;
        int io_comm_myrank, io_comm_nprocs;
        MPI_Datatype mpi_array_dtype;
    public:

        /* methods */
        field_binary_IO(
                const hsize_t *SIZES,
                const hsize_t *SUBSIZES,
                const hsize_t *STARTS,
                const MPI_Comm COMM_TO_USE);
        ~field_binary_IO();

        int read(
                const std::string fname,
                void *buffer);
        int write(
                const std::string fname,
                void *buffer);
};

#endif//FIELD_BINARY_IO_HPP

