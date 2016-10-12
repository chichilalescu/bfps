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



#include "fluid_solver_base.hpp"
#include "vorticity_equation.hpp"
#include "spline_n1.hpp"
#include "spline_n2.hpp"
#include "spline_n3.hpp"
#include "spline_n4.hpp"
#include "spline_n5.hpp"
#include "spline_n6.hpp"
#include "Lagrange_polys.hpp"

#ifndef INTERPOLATOR_BASE

#define INTERPOLATOR_BASE

typedef void (*base_polynomial_values)(
        const int derivative,
        const double fraction,
        double *__restrict__ destination);

template <class rnumber, int interp_neighbours>
class interpolator_base
{
    public:
        /* pointer to polynomial function */
        base_polynomial_values compute_beta;

        /* descriptor of field to interpolate */
        field_descriptor<rnumber> *descriptor;

        /* physical parameters of field */
        double dx, dy, dz;

        interpolator_base(
                fluid_solver_base<rnumber> *FSOLVER,
                base_polynomial_values BETA_POLYS);

        interpolator_base(
                vorticity_equation<rnumber, FFTW> *FSOLVER,
                base_polynomial_values BETA_POLYS);
        virtual ~interpolator_base(){}

        /* may not destroy input */
        virtual int read_rFFTW(const void *src) = 0;

        /* map real locations to grid coordinates */
        void get_grid_coordinates(
                const int nparticles,
                const int pdimension,
                const double *__restrict__ x,
                int *__restrict__ xg,
                double *__restrict__ xx);
        void get_grid_coordinates(
                const double *__restrict__ x,
                int *__restrict__ xg,
                double *__restrict__ xx);
        /* interpolate field at an array of locations */
        virtual void sample(
                const int nparticles,
                const int pdimension,
                const double *__restrict__ x,
                double *__restrict__ y,
                const int *deriv = NULL) = 0;
        /* interpolate 1 point */
        virtual void operator()(
                const int *__restrict__ xg,
                const double *__restrict__ xx,
                double *__restrict__ dest,
                const int *deriv = NULL) = 0;

        /* interpolate 1 point */
        inline void operator()(
                const double *__restrict__ x,
                double *__restrict__ dest,
                const int *deriv = NULL)
        {
            int xg[3];
            double xx[3];
            this->get_grid_coordinates(x, xg, xx);
            (*this)(xg, xx, dest, deriv);
        }
};

#endif//INTERPOLATOR_BASE

