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



#include "interpolator.hpp"

template <class rnumber>
interpolator<rnumber>::interpolator(
        fluid_solver_base<rnumber> *fs,
        const int bw)
{
    int tdims[4];
    this->buffer_width = bw;
    this->src_descriptor = fs->rd;
    this->buffer_size = this->buffer_width*this->src_descriptor->slice_size;
    tdims[0] = this->buffer_width*2*this->src_descriptor->nprocs + this->src_descriptor->sizes[0];
    tdims[1] = this->src_descriptor->sizes[1];
    tdims[2] = this->src_descriptor->sizes[2];
    tdims[3] = this->src_descriptor->sizes[3];
    this->descriptor = new field_descriptor<rnumber>(
            4, tdims,
            this->src_descriptor->mpi_dtype,
            this->src_descriptor->comm);
    this->f = new rnumber[this->descriptor->local_size];
    //if (sizeof(rnumber) == 4)
    //    this->f = fftwf_alloc_real(this->descriptor->local_size);
    //else if (sizeof(rnumber) == 8)
    //    this->f = fftw_alloc_real(this->descriptor->local_size);
}

template <class rnumber>
interpolator<rnumber>::~interpolator()
{
    delete[] this->f;
    delete this->descriptor;
}

template <class rnumber>
int interpolator<rnumber>::read_rFFTW(void *void_src)
{
    rnumber *src = (rnumber*)void_src;
    rnumber *dst = this->f;
    /* do big copy of middle stuff */
    std::copy(src,
              src + this->src_descriptor->local_size,
              dst + this->buffer_size);
    MPI_Datatype MPI_RNUM = (sizeof(rnumber) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    int rsrc;
    /* get upper slices */
    for (int rdst = 0; rdst < this->src_descriptor->nprocs; rdst++)
    {
        rsrc = this->src_descriptor->rank[(this->src_descriptor->all_start0[rdst] +
                                           this->src_descriptor->all_size0[rdst]) %
                                           this->src_descriptor->sizes[0]];
        if (this->src_descriptor->myrank == rsrc)
            MPI_Send(
                    src,
                    this->buffer_size,
                    MPI_RNUM,
                    rdst,
                    2*(rsrc*this->src_descriptor->nprocs + rdst),
                    this->descriptor->comm);
        if (this->src_descriptor->myrank == rdst)
            MPI_Recv(
                    dst + this->buffer_size + this->src_descriptor->local_size,
                    this->buffer_size,
                    MPI_RNUM,
                    rsrc,
                    2*(rsrc*this->src_descriptor->nprocs + rdst),
                    this->descriptor->comm,
                    MPI_STATUS_IGNORE);
    }
    /* get lower slices */
    for (int rdst = 0; rdst < this->src_descriptor->nprocs; rdst++)
    {
        rsrc = this->src_descriptor->rank[MOD(this->src_descriptor->all_start0[rdst] - 1,
                                              this->src_descriptor->sizes[0])];
        if (this->src_descriptor->myrank == rsrc)
            MPI_Send(
                    src + this->src_descriptor->local_size - this->buffer_size,
                    this->buffer_size,
                    MPI_RNUM,
                    rdst,
                    2*(rsrc*this->src_descriptor->nprocs + rdst)+1,
                    this->src_descriptor->comm);
        if (this->src_descriptor->myrank == rdst)
            MPI_Recv(
                    dst,
                    this->buffer_size,
                    MPI_RNUM,
                    rsrc,
                    2*(rsrc*this->src_descriptor->nprocs + rdst)+1,
                    this->src_descriptor->comm,
                    MPI_STATUS_IGNORE);
    }
    return EXIT_SUCCESS;
}

template class interpolator<float>;
template class interpolator<double>;

