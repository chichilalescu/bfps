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



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hdf5.h>
#include "base.hpp"
#include "particles_base.hpp"
#include "fluid_solver_base.hpp"
#include "interpolator.hpp"

#ifndef PARTICLES

#define PARTICLES

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
class particles
{
    public:
        int myrank, nprocs;
        MPI_Comm comm;
        fluid_solver_base<rnumber> *fs;
        rnumber *data;

        /* watching is an array of shape [nparticles], with
         * watching[p] being true if particle p is in the domain of myrank
         * or in the buffer regions.
         * only used if multistep is false.
         * */
        bool *watching;
        /* computing is an array of shape [nparticles], with
         * computing[p] being the rank that is currently working on particle p
         * */
        int *computing;

        /* state will generally hold all the information about the particles.
         * in the beginning, we will only need to solve 3D ODEs, but I figured
         * a general ncomponents is better, since we may change our minds.
         * */
        double *state;
        double *rhs[6];
        int nparticles;
        int ncomponents;
        int array_size;
        int integration_steps;
        int traj_skip;
        int buffer_width;
        ptrdiff_t buffer_size;
        double *lbound;
        double *ubound;
        interpolator<rnumber, interp_neighbours> *vel;

        /* simulation parameters */
        char name[256];
        int iteration;
        double dt;

        /* physical parameters of field */
        double dx, dy, dz;

        /* methods */

        /* constructor and destructor.
         * allocate and deallocate:
         *  this->state
         *  this->rhs
         *  this->lbound
         *  this->ubound
         *  this->computing
         *  this->watching
         * */
        particles(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER,
                interpolator<rnumber, interp_neighbours> *FIELD,
                const int NPARTICLES,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS = 2);
        ~particles();

        /* an Euler step is needed to compute an estimate of future positions,
         * which is needed for synchronization.
         * */
        void jump_estimate(double *__restrict__ jump_length);
        void get_rhs(double *__restrict__ x, double *__restrict__ rhs);
        void get_rhs(double t, double *__restrict__ x, double *__restrict__ rhs);

        int get_rank(double z); // get rank for given value of z
        void synchronize();
        void synchronize_single_particle_state(int p, double *__restrict__ x, int source_id = -1);
        void get_grid_coordinates(double *__restrict__ x, int *__restrict__ xg, double *__restrict__ xx);
        void sample_vec_field(
            interpolator<rnumber, interp_neighbours> *vec,
            double t,
            double *__restrict__ x,
            double *__restrict__ y,
            const bool synch = false,
            int *deriv = NULL);
        inline void sample_vec_field(interpolator<rnumber, interp_neighbours> *field, double *vec_values)
        {
            this->sample_vec_field(field, 1.0, this->state, vec_values, true, NULL);
        }

        /* input/output */
        void read(hid_t data_file_id);
        void write(hid_t data_file_id, bool write_rhs = true);

        /* solvers */
        void step();
        void roll_rhs();
        void AdamsBashforth(int nsteps);
        void Heun();
        void cRK4();
};

#endif//PARTICLES

