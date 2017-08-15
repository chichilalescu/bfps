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



#include "field_descriptor.hpp"
#include "fftw_tools.hpp"
#include "fluid_solver_base.hpp"
#include "vorticity_equation.hpp"
#include "interpolator_base.hpp"

#ifndef RFFTW_INTERPOLATOR

#define RFFTW_INTERPOLATOR

template <class rnumber, int interp_neighbours>
class rFFTW_interpolator:public interpolator_base<rnumber, interp_neighbours>
{
    public:
        using interpolator_base<rnumber, interp_neighbours>::operator();

        /* pointer to field that has to be interpolated
         * The reason this is a member variable is because I want this class
         * to be consistent with the "interpolator" class, where a member
         * variable is absolutely required (since that class uses padding).
         * */
        rnumber *field;

        /* compute[iz] is an array that says whether or not the current MPI
         * process is involved in the interpolation formula for a particle
         * located in cell "iz".
         * It is mostly used in the formula itself.
         * This translates as the following condition:
         * local_zstart - neighbours <= iz <= local_zend + 1 + neighbours
         * I think it's cleaner to keep things in an array, especially since
         * "local_zend" is shorthand for another arithmetic operation anyway.
         * */
        bool *compute;


        /* Constructors */
        rFFTW_interpolator(
                fluid_solver_base<rnumber> *FSOLVER,
                base_polynomial_values BETA_POLYS,
                rnumber *FIELD_DATA);

        /* this constructor is empty, I just needed for a quick hack of the
         * "vorticity_equation" class.
         * It should be removed soon.
         * */
        rFFTW_interpolator(
                vorticity_equation<rnumber, FFTW> *FSOLVER,
                base_polynomial_values BETA_POLYS,
                rnumber *FIELD_DATA);
        ~rFFTW_interpolator();

        /* This method is provided for consistency with "interpolator", and it
         * does not destroy input */
        inline int read_rFFTW(const void *src)
        {
            this->field = (rnumber*)src;
            return EXIT_SUCCESS;
        }

        /* This is used when "compute" is not enough.
         * For a given z location, it gives the outermost ranks that are relevant
         * for the interpolation formula.
         * */
        bool get_rank_info(double z, int &maxz_rank, int &minz_rank);

        /* interpolate field at an array of locations.
         * After interpolation is performed, call Allreduce for "y", over
         * this->descriptor->comm --- generally MPI_COMM_WORLD.
         * This is useful for the simple "particles" class, where particle
         * information is synchronized across all processes.
         * */
        void sample(
                const int nparticles,
                const int pdimension,
                const double *__restrict__ x,
                double *__restrict__ y,
                const int *deriv = NULL);
        /* interpolate 1 point.
         * Result is kept local.
         * This is used in the "rFFTW_distributed_particles" class, with the
         * result being synchronized across the relevant "local particle
         * communicator".
         * */
        void operator()(
                const int *__restrict__ xg,
                const double *__restrict__ xx,
                double *__restrict__ dest,
                const int *deriv = NULL);
};

#endif//RFFTW_INTERPOLATOR

