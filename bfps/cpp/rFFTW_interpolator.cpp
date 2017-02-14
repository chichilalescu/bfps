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

#include <cmath>
#include "rFFTW_interpolator.hpp"
#include "scope_timer.hpp"

template <class rnumber, int interp_neighbours>
rFFTW_interpolator<rnumber, interp_neighbours>::rFFTW_interpolator(
        fluid_solver_base<rnumber> *fs,
        base_polynomial_values BETA_POLYS,
        rnumber *FIELD_POINTER) : interpolator_base<rnumber, interp_neighbours>(fs, BETA_POLYS)
{
    this->field = FIELD_POINTER;


    // generate compute array
    this->compute = new bool[this->descriptor->sizes[0]];
    std::fill_n(this->compute, this->descriptor->sizes[0], false);
    for (int iz = this->descriptor->starts[0]-interp_neighbours-1;
            iz <= this->descriptor->starts[0]+this->descriptor->subsizes[0]+interp_neighbours;
            iz++)
        this->compute[((iz + this->descriptor->sizes[0]) % this->descriptor->sizes[0])] = true;
}

template <class rnumber, int interp_neighbours>
rFFTW_interpolator<rnumber, interp_neighbours>::rFFTW_interpolator(
        vorticity_equation<rnumber, FFTW> *fs,
        base_polynomial_values BETA_POLYS,
        rnumber *FIELD_POINTER) : interpolator_base<rnumber, interp_neighbours>(fs, BETA_POLYS)
{
//    this->field = FIELD_POINTER;
//
//
//    // generate compute array
//    this->compute = new bool[this->descriptor->sizes[0]];
//    std::fill_n(this->compute, this->descriptor->sizes[0], false);
//    for (int iz = this->descriptor->starts[0]-interp_neighbours-1;
//            iz <= this->descriptor->starts[0]+this->descriptor->subsizes[0]+interp_neighbours;
//            iz++)
//        this->compute[((iz + this->descriptor->sizes[0]) % this->descriptor->sizes[0])] = true;
}

template <class rnumber, int interp_neighbours>
rFFTW_interpolator<rnumber, interp_neighbours>::~rFFTW_interpolator()
{
    delete[] this->compute;
}

template <class rnumber, int interp_neighbours>
bool rFFTW_interpolator<rnumber, interp_neighbours>::get_rank_info(double z, int &maxz_rank, int &minz_rank)
{
    int zg = int(floor(z/this->dz));
    minz_rank = this->descriptor->rank[MOD(
             zg - interp_neighbours,
            this->descriptor->sizes[0])];
    maxz_rank = this->descriptor->rank[MOD(
            zg + 1 + interp_neighbours,
            this->descriptor->sizes[0])];
    bool is_here = false;
    for (int iz = -interp_neighbours; iz <= interp_neighbours+1; iz++)
        is_here = (is_here ||
                   (this->descriptor->myrank ==
                    this->descriptor->rank[MOD(zg+iz, this->descriptor->sizes[0])]));
    return is_here;
}

template <class rnumber, int interp_neighbours>
void rFFTW_interpolator<rnumber, interp_neighbours>::sample(
        const int nparticles,
        const int pdimension,
        const double *__restrict__ x,
        double *__restrict__ y,
        const int *deriv)
{
    TIMEZONE("rFFTW_interpolator::sample");
    /* get grid coordinates */
    int *xg = new int[3*nparticles];
    double *xx = new double[3*nparticles];
    double *yy =  new double[3*nparticles];
    std::fill_n(yy, 3*nparticles, 0.0);
    this->get_grid_coordinates(nparticles, pdimension, x, xg, xx);
    /* perform interpolation */
    for (int p=0; p<nparticles; p++)
        if (this->compute[xg[p*3+2]])
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
void rFFTW_interpolator<rnumber, interp_neighbours>::operator()(
        const int *xg,
        const double *xx,
        double *dest,
        const int *deriv)
{
    TIMEZONE("rFFTW_interpolator::operator()");
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
        bigiz = ptrdiff_t(((xg[2]+iz) + this->descriptor->sizes[0]) % this->descriptor->sizes[0]);
        if (this->descriptor->myrank == this->descriptor->rank[bigiz])
        {
            for (int iy = -interp_neighbours; iy <= interp_neighbours+1; iy++)
            {
                bigiy = ptrdiff_t(MOD(xg[1]+iy, this->descriptor->sizes[1]));
                for (int ix = -interp_neighbours; ix <= interp_neighbours+1; ix++)
                {
                    bigix = ptrdiff_t(MOD(xg[0]+ix, this->descriptor->sizes[2]));
                    ptrdiff_t tindex = (((bigiz-this->descriptor->starts[0])*this->descriptor->sizes[1] +
                                         bigiy)*(this->descriptor->sizes[2]+2) +
                                         bigix)*3;
                    for (int c=0; c<3; c++)
                        dest[c] += this->field[tindex+c]*(bz[iz+interp_neighbours]*
                                                          by[iy+interp_neighbours]*
                                                          bx[ix+interp_neighbours]);
                }
            }
        }
    }
}

template class rFFTW_interpolator<float, 1>;
template class rFFTW_interpolator<float, 2>;
template class rFFTW_interpolator<float, 3>;
template class rFFTW_interpolator<float, 4>;
template class rFFTW_interpolator<float, 5>;
template class rFFTW_interpolator<float, 6>;
template class rFFTW_interpolator<double, 1>;
template class rFFTW_interpolator<double, 2>;
template class rFFTW_interpolator<double, 3>;
template class rFFTW_interpolator<double, 4>;
template class rFFTW_interpolator<double, 5>;
template class rFFTW_interpolator<double, 6>;

