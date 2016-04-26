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


#include "field.hpp"

template <class number,
          field_backend be,
          field_components fc>
field_container<number, be, fc>::field_container(
        int nx, int ny, int nz,
        MPI_Comm COMM_TO_USE)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);

    if (be == FFTW)
    {
        ptrdiff_t nfftw[3];
        nfftw[0] = nz;
        nfftw[1] = ny;
        nfftw[2] = nx;
        this->local_size = fftw_mpi_local_size_many(
                3,
                nfftw,
                ncomp(fc),
                FFTW_MPI_DEFAULT_BLOCK,
                this->comm,
                this->subsizes,
                this->starts);
    }
    this->sizes[0] = nz;
    this->sizes[1] = ny;
    this->sizes[2] = nx;
    if (fc == THREE || fc == THREExTHREE)
        this->sizes[3] = 3;
    if (fc == THREExTHREE)
        this->sizes[4] = 3;
    if (be == FFTW)
    {
        for (int i=1; i<ndim(fc); i++)
        {
            this->subsizes[i] = this->sizes[i];
            this->starts[i] = 0;
        }
    }
}

template <class number,
          field_backend be,
          field_components fc>
field_container<number, be, fc>::~field_container()
{}

template <class number,
          field_backend be,
          field_components fc>
int field_container<number, be, fc>::read(
        const char *fname,
        const char *dset_name,
        int iteration)
{
    return EXIT_SUCCESS;
}

template <class number,
          field_backend be,
          field_components fc>
int field_container<number, be, fc>::write(
        const char *fname,
        const char *dset_name,
        int iteration)
{
    return EXIT_SUCCESS;
}

//template <class rnum,
//          field_representation repr,
//          field_backend be,
//          field_components fc>
//field::field(
//                int *n,
//                MPI_Comm COMM_TO_USE)
//{}
//
//field::~field()
//{}

