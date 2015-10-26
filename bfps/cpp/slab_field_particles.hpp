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

#ifndef SLAB_FIELD_PARTICLES

#define SLAB_FIELD_PARTICLES

extern int myrank, nprocs;

template <class rnumber>
class slab_field_particles
{
    protected:
        //typedef void (slab_field_particles<rnumber>::*tensor_product_interpolation_formula)(
        //        rnumber *field,
        //        int *xg,
        //        double *xx,
        //        double *dest,
        //        int *deriv);
        typedef void (*base_polynomial_values)(
                int derivative,
                double fraction,
                double *destination);
    public:
        fluid_solver_base<rnumber> *fs;
        field_descriptor<rnumber> *buffered_field_descriptor;

        /* watching is an array of shape [nparticles], with
         * watching[p] being true if particle p is in the domain of myrank
         * or in the buffer regions.
         * watching is not really being used right now, since I don't do partial
         * synchronizations of particles.
         * we may do this at some point in the future, if it seems needed...
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
        int interp_neighbours;
        int interp_smoothness;
        int buffer_width;
        int integration_steps;
        int traj_skip;
        ptrdiff_t buffer_size;
        double *lbound;
        double *ubound;
        //tensor_product_interpolation_formula spline_formula;
        base_polynomial_values compute_beta;

        /* simulation parameters */
        char name[256];
        int iteration;
        double dt;

        /* physical parameters of field */
        rnumber dx, dy, dz;

        /* methods */

        /* constructor and destructor.
         * allocate and deallocate:
         *  this->state
         *  this->lbound
         *  this->ubound
         *  this->watching
         * */
        slab_field_particles(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER,
                const int NPARTICLES,
                const int NCOMPONENTS,
                const int INTERP_NEIGHBOURS,
                const int INTERP_SMOOTHNESS,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS = 2);
        ~slab_field_particles();

        /* an Euler step is needed to compute an estimate of future positions,
         * which is needed for synchronization.
         * */
        virtual void jump_estimate(double *jump_length);
        /* function get_rhs is virtual since we want children to do different things,
         * depending on the type of particle.
         * */
        virtual void get_rhs(double *x, double *rhs);

        /* generic methods, should work for all children of this class */
        int get_rank(double z); // get rank for given value of z
        void synchronize();
        void synchronize_single_particle(int p);
        void get_grid_coordinates(double *x, int *xg, double *xx);
        void linear_interpolation(rnumber *field, int *xg, double *xx, double *dest, int *deriv);
        void spline_formula(      rnumber *field, int *xg, double *xx, double *dest, int *deriv);
        void spline_n1_formula(   rnumber *field, int *xg, double *xx, double *dest, int *deriv);
        void spline_n2_formula(   rnumber *field, int *xg, double *xx, double *dest, int *deriv);
        void spline_n3_formula(   rnumber *field, int *xg, double *xx, double *dest, int *deriv);

        void rFFTW_to_buffered(rnumber *src, rnumber *dst);

        /* generic methods, should work for all children of this class */
        void read(hid_t data_file_id);
        void write(hid_t data_file_id, bool write_rhs = true);

        /* solver stuff */
        void step();
        void roll_rhs();
        void AdamsBashforth(int nsteps);
        void Euler();
};


#endif//SLAB_FIELD_PARTICLES

