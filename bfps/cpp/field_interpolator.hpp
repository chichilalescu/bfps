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
#include "interpolator_base.hpp"

#ifndef FIELD_INTERPOLATOR

#define FIELD_INTERPOLATOR

template <class rnumber,
          int interp_neighbours,
          field_backend be,
          field_components fc>
class field_interpolator:
{
    public:

        field_layout<fc> *flayout;

        /*
         * for FFTW:
         * compute[iz] is true if
         * local_zstart - neighbours <= iz <= local_zend + 1 + neighbours
         *
         * Generally "compute" should say whether or not the current CPU
         * needs to perform the computation for the sample point being in
         * a certain location.
         * I assume it will be a 2D array when 2D decomposition is used.
         * I've no idea what will happen when using MPI-OpenMP hybrid.
         */
        bool *compute;

        field_interpolator(
                fluid_solver_base<rnumber> *FSOLVER,
                base_polynomial_values BETA_POLYS);
        ~field_interpolator();

        bool get_rank_info(double z, int &maxz_rank, int &minz_rank);

        /* interpolate field at an array of locations */
        void sample(
                field<rnumber, be, fc> *f,
                const int nparticles,
                const int pdimension,
                const double *__restrict__ x,
                double *__restrict__ y,
                const int *deriv = NULL);
        /* interpolate 1 point */
        void operator()(
                field<rnumber, be, fc> *f,
                const int *__restrict__ xg,
                const double *__restrict__ xx,
                double *__restrict__ dest,
                const int *deriv = NULL);
};

#endif//FIELD_INTERPOLATOR

