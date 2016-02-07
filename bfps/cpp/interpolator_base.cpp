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
#include "interpolator_base.hpp"

template <class rnumber, int interp_neighbours>
interpolator_base<rnumber, interp_neighbours>::interpolator_base(
        fluid_solver_base<rnumber> *fs,
        base_polynomial_values BETA_POLYS)
{
    this->descriptor = fs->rd;
    this->compute_beta = BETA_POLYS;

    // compute dx, dy, dz;
    this->dx = 4*acos(0) / (fs->dkx*this->descriptor->sizes[2]);
    this->dy = 4*acos(0) / (fs->dky*this->descriptor->sizes[1]);
    this->dz = 4*acos(0) / (fs->dkz*this->descriptor->sizes[0]);
}

template <class rnumber, int interp_neighbours>
void interpolator_base<rnumber, interp_neighbours>::get_grid_coordinates(
        const int nparticles,
        const int pdimension,
        const double *x,
        int *xg,
        double *xx)
{
    static double grid_size[] = {this->dx, this->dy, this->dz};
    double tval;
    std::fill_n(xg, nparticles*3, 0);
    std::fill_n(xx, nparticles*3, 0.0);
    for (int p=0; p<nparticles; p++)
    {
        for (int c=0; c<3; c++)
        {
            tval = floor(x[p*pdimension+c]/grid_size[c]);
            xg[p*3+c] = MOD(int(tval), this->descriptor->sizes[2-c]);
            xx[p*3+c] = (x[p*pdimension+c] - tval*grid_size[c]) / grid_size[c];
        }
    }
}

template class interpolator_base<float, 1>;
template class interpolator_base<float, 2>;
template class interpolator_base<float, 3>;
template class interpolator_base<float, 4>;
template class interpolator_base<float, 5>;
template class interpolator_base<float, 6>;
template class interpolator_base<double, 1>;
template class interpolator_base<double, 2>;
template class interpolator_base<double, 3>;
template class interpolator_base<double, 4>;
template class interpolator_base<double, 5>;
template class interpolator_base<double, 6>;

