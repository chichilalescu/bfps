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



#include "slab_field_particles.hpp"

#ifndef TRACERS

#define TRACERS

extern int myrank, nprocs;

template <class rnumber>
class tracers final:public slab_field_particles<rnumber>
{
    public:
        rnumber *source_data;
        rnumber *data;

        /* methods */
        tracers(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER,
                const int NPARTICLES,
                base_polynomial_values BETA_POLYS,
                const int NEIGHBOURS,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS,
                rnumber *SOURCE_DATA);
        ~tracers();

        void update_field(bool clip_on = true);
        virtual void get_rhs(double *x, double *rhs);
        virtual void jump_estimate(double *jump_length);

        void sample_vec_field(rnumber *vec_field, double *vec_values);
};


#endif//TRACERS

