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



#include <mpi.h>
#include <fftw3-mpi.h>

#ifndef FIELD_DESCRIPTOR

#define FIELD_DESCRIPTOR

extern int myrank, nprocs;

template <class rnumber>
class field_descriptor
{
    private:
        typedef rnumber cnumber[2];
    public:

        /* data */
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int *rank;
        int *all_start0;
        int *all_size0;
        ptrdiff_t slice_size, local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;
        int myrank, nprocs, io_myrank, io_nprocs;
        MPI_Comm comm, io_comm;


        /* methods */
        field_descriptor(
                int ndims,
                int *n,
                MPI_Datatype element_type,
                MPI_Comm COMM_TO_USE);
        ~field_descriptor();

        /* io is performed using MPI_File stuff, and our
         * own mpi_array_dtype that was defined in the constructor.
         * */
        int read(
                const char *fname,
                void *buffer);
        int write(
                const char *fname,
                void *buffer);

        /* a function that generates the transposed descriptor.
         * don't forget to delete the result once you're done with it.
         * the transposed descriptor is useful for io operations.
         * */
        field_descriptor<rnumber> *get_transpose();

        /* we don't actually need the transposed descriptor to perform
         * the transpose operation: we only need the in/out fields.
         * */
        int transpose(
                rnumber *input,
                rnumber *output);
        int transpose(
                cnumber *input,
                cnumber *output = NULL);

        int interleave(
                rnumber *input,
                int dim);
        int interleave(
                cnumber *input,
                int dim);
};


inline float btle(const float be)
     {
         float le;
         char *befloat = (char *) & be;
         char *lefloat = (char *) & le;
         lefloat[0] = befloat[3];
         lefloat[1] = befloat[2];
         lefloat[2] = befloat[1];
         lefloat[3] = befloat[0];
         return le;
     }

#endif//FIELD_DESCRIPTOR

