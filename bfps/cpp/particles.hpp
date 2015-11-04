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
#include "fluid_solver_base.hpp"
#include "spline_n1.hpp"
#include "spline_n2.hpp"
#include "spline_n3.hpp"
#include "spline_n4.hpp"
#include "spline_n5.hpp"
#include "spline_n6.hpp"
#include "Lagrange_polys.hpp"

#ifndef PARTICLES

#define PARTICLES

extern int myrank, nprocs;

typedef void (*base_polynomial_values)(
        int derivative,
        double fraction,
        double *destination);

/* particle types */
enum particle_types {VELOCITY_TRACER};

template <int particle_type, class rnumber, bool multistep, int ncomponents, int interp_neighbours>
class particles
{
    public:
        fluid_solver_base<rnumber> *fs;
        field_descriptor<rnumber> *buffered_field_descriptor;
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
        int array_size;
        int integration_steps;
        int traj_skip;
        int buffer_width;
        ptrdiff_t buffer_size;
        double *lbound;
        double *ubound;
        base_polynomial_values compute_beta;

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
                const int NPARTICLES,
                base_polynomial_values BETA_POLYS,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS = 2);
        ~particles();
        void rFFTW_to_buffered(float *src, float *dst);
        void rFFTW_to_buffered(double *src, double *dst);

        /* an Euler step is needed to compute an estimate of future positions,
         * which is needed for synchronization.
         * */
        void jump_estimate(double *jump_length);
        void get_rhs(double *x, double *rhs);

        int get_rank(double z); // get rank for given value of z
        void synchronize();
        void synchronize_single_particle_state(int p, double *x, int source_id = -1);
        void get_grid_coordinates(double *x, int *xg, double *xx);
        void interpolation_formula(rnumber *field, int *xg, double *xx, double *dest, int *deriv);


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

