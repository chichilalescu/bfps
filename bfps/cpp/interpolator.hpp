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



#include <cmath>
#include "field_descriptor.hpp"
#include "fftw_tools.hpp"
#include "fluid_solver_base.hpp"
#include "interpolator_base.hpp"

#ifndef INTERPOLATOR

#define INTERPOLATOR

template <class rnumber, int interp_neighbours>
class interpolator:public interpolator_base<rnumber, interp_neighbours>
{
    public:
        ptrdiff_t buffer_size;

        /* descriptor for buffered field */
        field_descriptor<rnumber> *buffered_descriptor;

        /* pointer to buffered field */
        rnumber *field;

        interpolator(
                fluid_solver_base<rnumber> *FSOLVER,
                base_polynomial_values BETA_POLYS,
                ...);
        ~interpolator();

        int read_rFFTW(const void *src);

        inline int get_rank(double z)
        {
            return this->descriptor->rank[MOD(int(floor(z/this->dz)), this->descriptor->sizes[0])];
        }

        /* interpolate field at an array of locations */
        void sample(
                const int nparticles,
                const int pdimension,
                const double *__restrict__ x,
                double *__restrict__ y,
                const int *deriv = NULL);
        /* interpolate 1 point */
        void operator()(
                const int *__restrict__ xg,
                const double *__restrict__ xx,
                double *__restrict__ dest,
                const int *deriv = NULL);
};

#endif//INTERPOLATOR

