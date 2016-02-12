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



#define NDEBUG

#include "interpolator.hpp"

template <class rnumber, int interp_neighbours>
interpolator<rnumber, interp_neighbours>::interpolator(
        fluid_solver_base<rnumber> *fs,
        base_polynomial_values BETA_POLYS,
        ...) : interpolator_base<rnumber, interp_neighbours>(fs, BETA_POLYS)
{
    int tdims[4];
    this->compute_beta = BETA_POLYS;
    tdims[0] = (interp_neighbours+1)*2*this->descriptor->nprocs + this->descriptor->sizes[0];
    tdims[1] = this->descriptor->sizes[1];
    tdims[2] = this->descriptor->sizes[2]+2;
    tdims[3] = this->descriptor->sizes[3];
    this->buffered_descriptor = new field_descriptor<rnumber>(
            4, tdims,
            this->descriptor->mpi_dtype,
            this->descriptor->comm);
    this->buffer_size = (interp_neighbours+1)*this->buffered_descriptor->slice_size;
    this->field = new rnumber[this->buffered_descriptor->local_size];
}

template <class rnumber, int interp_neighbours>
interpolator<rnumber, interp_neighbours>::~interpolator()
{
    delete[] this->field;
    delete this->buffered_descriptor;
}

template <class rnumber, int interp_neighbours>
int interpolator<rnumber, interp_neighbours>::read_rFFTW(const void *void_src)
{
    rnumber *src = (rnumber*)void_src;
    rnumber *dst = this->field;
    /* do big copy of middle stuff */
    std::copy(src,
              src + this->buffered_descriptor->slice_size*this->descriptor->subsizes[0],
              dst + this->buffer_size);
    MPI_Datatype MPI_RNUM = (sizeof(rnumber) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    int rsrc;
    /* get upper slices */
    for (int rdst = 0; rdst < this->descriptor->nprocs; rdst++)
    {
        rsrc = this->descriptor->rank[(this->descriptor->all_start0[rdst] +
                                       this->descriptor->all_size0[rdst]) %
                                       this->descriptor->sizes[0]];
        if (this->descriptor->myrank == rsrc)
            MPI_Send(
                    src,
                    this->buffer_size,
                    MPI_RNUM,
                    rdst,
                    2*(rsrc*this->descriptor->nprocs + rdst),
                    this->buffered_descriptor->comm);
        if (this->descriptor->myrank == rdst)
            MPI_Recv(
                    dst + this->buffer_size + this->buffered_descriptor->slice_size*this->descriptor->subsizes[0],
                    this->buffer_size,
                    MPI_RNUM,
                    rsrc,
                    2*(rsrc*this->descriptor->nprocs + rdst),
                    this->buffered_descriptor->comm,
                    MPI_STATUS_IGNORE);
    }
    /* get lower slices */
    for (int rdst = 0; rdst < this->descriptor->nprocs; rdst++)
    {
        rsrc = this->descriptor->rank[MOD(this->descriptor->all_start0[rdst] - 1,
                                          this->descriptor->sizes[0])];
        if (this->descriptor->myrank == rsrc)
            MPI_Send(
                    src + this->buffered_descriptor->slice_size*this->descriptor->subsizes[0] - this->buffer_size,
                    this->buffer_size,
                    MPI_RNUM,
                    rdst,
                    2*(rsrc*this->descriptor->nprocs + rdst)+1,
                    this->descriptor->comm);
        if (this->descriptor->myrank == rdst)
            MPI_Recv(
                    dst,
                    this->buffer_size,
                    MPI_RNUM,
                    rsrc,
                    2*(rsrc*this->descriptor->nprocs + rdst)+1,
                    this->descriptor->comm,
                    MPI_STATUS_IGNORE);
    }
    return EXIT_SUCCESS;
}

template <class rnumber, int interp_neighbours>
void interpolator<rnumber, interp_neighbours>::sample(
        const int nparticles,
        const int pdimension,
        const double *__restrict__ x,
        double *__restrict__ y,
        const int *deriv)
{
    /* get grid coordinates */
    int *xg = new int[3*nparticles];
    double *xx = new double[3*nparticles];
    double *yy = new double[3*nparticles];
    std::fill_n(yy, 3*nparticles, 0.0);
    this->get_grid_coordinates(nparticles, pdimension, x, xg, xx);
    /* perform interpolation */
    for (int p=0; p<nparticles; p++)
        if (this->descriptor->rank[MOD(xg[p*3+2], this->descriptor->sizes[0])] == this->descriptor->myrank)
            this->operator()(xg + p*3, xx + p*3, yy + p*3, deriv);
    MPI_Allreduce(
            yy,
            y,
            3*nparticles,
            MPI_DOUBLE,
            MPI_SUM,
            this->descriptor->comm);
    delete[] yy;
    delete[] xg;
    delete[] xx;
}

template <class rnumber, int interp_neighbours>
void interpolator<rnumber, interp_neighbours>::operator()(
        const int *xg,
        const double *xx,
        double *dest,
        const int *deriv)
{
    double bx[interp_neighbours*2+2], by[interp_neighbours*2+2], bz[interp_neighbours*2+2];
    if (deriv == NULL)
    {
        this->compute_beta(0, xx[0], bx);
        this->compute_beta(0, xx[1], by);
        this->compute_beta(0, xx[2], bz);
    }
    else
    {
        this->compute_beta(deriv[0], xx[0], bx);
        this->compute_beta(deriv[1], xx[1], by);
        this->compute_beta(deriv[2], xx[2], bz);
    }
    std::fill_n(dest, 3, 0);
    ptrdiff_t bigiz, bigiy, bigix;
    for (int iz = -interp_neighbours; iz <= interp_neighbours+1; iz++)
    {
        bigiz = ptrdiff_t(xg[2]+iz)-this->descriptor->starts[0];
        for (int iy = -interp_neighbours; iy <= interp_neighbours+1; iy++)
        {
            bigiy = ptrdiff_t(MOD(xg[1]+iy, this->descriptor->sizes[1]));
            for (int ix = -interp_neighbours; ix <= interp_neighbours+1; ix++)
            {
                bigix = ptrdiff_t(MOD(xg[0]+ix, this->descriptor->sizes[2]));
                ptrdiff_t tindex = ((bigiz *this->buffered_descriptor->sizes[1] +
                                     bigiy)*this->buffered_descriptor->sizes[2] +
                                     bigix)*3 + this->buffer_size;
                for (int c=0; c<3; c++)
                {
                    dest[c] += this->field[tindex+c]*(bz[iz+interp_neighbours]*
                                                      by[iy+interp_neighbours]*
                                                      bx[ix+interp_neighbours]);
                }
            }
        }
    }
}

template class interpolator<float, 1>;
template class interpolator<float, 2>;
template class interpolator<float, 3>;
template class interpolator<float, 4>;
template class interpolator<float, 5>;
template class interpolator<float, 6>;
template class interpolator<double, 1>;
template class interpolator<double, 2>;
template class interpolator<double, 3>;
template class interpolator<double, 4>;
template class interpolator<double, 5>;
template class interpolator<double, 6>;

