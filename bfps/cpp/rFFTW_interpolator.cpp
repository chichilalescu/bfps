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



#include "rFFTW_interpolator.hpp"

template <class rnumber, int interp_neighbours>
rFFTW_interpolator<rnumber, interp_neighbours>::rFFTW_interpolator(
        fluid_solver_base<rnumber> *fs,
        base_polynomial_values BETA_POLYS)
{
    this->descriptor = fs->rd;
    this->field_size = 2*fs->cd->local_size;
    this->compute_beta = BETA_POLYS;
    if (sizeof(rnumber) == 4)
    {
        this->f0 = (rnumber*)((void*)fftwf_alloc_real(this->field_size));
        this->f1 = (rnumber*)((void*)fftwf_alloc_real(this->field_size));
    }
    else if (sizeof(rnumber) == 8)
    {
        this->f0 = (rnumber*)((void*)fftw_alloc_real(this->field_size));
        this->f1 = (rnumber*)((void*)fftw_alloc_real(this->field_size));
    }
    this->temp = this->f1;
}

template <class rnumber, int interp_neighbours>
rFFTW_interpolator<rnumber, interp_neighbours>::~rFFTW_interpolator()
{
    if (sizeof(rnumber) == 4)
    {
        fftwf_free((float*)((void*)this->f0));
        fftwf_free((float*)((void*)this->f1));
    }
    else if (sizeof(rnumber) == 8)
    {
        fftw_free((double*)((void*)this->f0));
        fftw_free((double*)((void*)this->f1));
    }
}

template <class rnumber, int interp_neighbours>
int rFFTW_interpolator<rnumber, interp_neighbours>::read_rFFTW(void *void_src)
{
    /* first, roll fields */
    rnumber *tmp = this->f0;
    this->f0 = this->f1;
    this->f1 = tmp;
    this->temp = this->f0;
    /* now do regular things */
    rnumber *src = (rnumber*)void_src;
    rnumber *dst = this->f1;
    /* do big copy of middle stuff */
    std::copy(src,
              src + this->field_size,
              dst);
    return EXIT_SUCCESS;
}

template <class rnumber, int interp_neighbours>
void rFFTW_interpolator<rnumber, interp_neighbours>::operator()(
        double t,
        int *xg,
        double *xx,
        double *dest,
        int *deriv)
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
    double tval[3];
    for (int iz = -interp_neighbours; iz <= interp_neighbours+1; iz++)
    {
        bigiz = ptrdiff_t(((xg[2]+iz) + this->descriptor->sizes[0]) % this->descriptor->sizes[0]);
        if (this->descriptor->myrank == this->descriptor->rank[bigiz])
        {
            std::fill_n(tval, 3, 0);
            for (int iy = -interp_neighbours; iy <= interp_neighbours+1; iy++)
            {
                bigiy = ptrdiff_t(MOD(xg[1]+iy, this->descriptor->sizes[1]));
                for (int ix = -interp_neighbours; ix <= interp_neighbours+1; ix++)
                {
                    bigix = ptrdiff_t(MOD(xg[0]+ix, this->descriptor->sizes[2]));
                    for (int c=0; c<3; c++)
                    {
                        ptrdiff_t tindex = ((bigiz *this->descriptor->sizes[1] +
                                             bigiy)*(this->descriptor->sizes[2]+2) +
                                             bigix)*3+c;
                        dest[c] += (this->f0[tindex]*(1-t) + t*this->f1[tindex])*(bz[iz+interp_neighbours]*
                                                                                  by[iy+interp_neighbours]*
                                                                                  bx[ix+interp_neighbours]);
                        tval[c] += (this->f0[tindex]*(1-t) + t*this->f1[tindex])*(bz[iz+interp_neighbours]*
                                                                                  by[iy+interp_neighbours]*
                                                                                  bx[ix+interp_neighbours]);
                    }
                }
            }
            DEBUG_MSG("%ld %d %d %g %g %g\n", bigiz, xg[1], xg[0], tval[0], tval[1], tval[2]);
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

