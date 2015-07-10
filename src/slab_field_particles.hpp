/***********************************************************************
*
*  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: Cristian.Lalescu@ds.mpg.de
*
************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "base.hpp"
#include "fluid_solver_base.hpp"

#ifndef SLAB_FIELD_PARTICLES

#define SLAB_FIELD_PARTICLES

extern int myrank, nprocs;

template <class rnumber>
class slab_field_particles
{
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
        int nparticles;
        int ncomponents;
        int array_size;
        int buffer_size;
        double *lbound;
        double *ubound;

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
                const int BUFFERSIZE);
        ~slab_field_particles();

        /* an Euler step is needed to compute an estimate of future positions,
         * which is needed for synchronization.
         * functions are virtual since we want children to do different things,
         * depending on the type of particle.
         * */
        virtual void jump_estimate(double *jump_length);
        virtual void get_rhs(double *x, double *rhs);

        /* generic methods, should work for all children of this class */
        int get_rank(double z); // get rank for given value of z
        void synchronize();
        void synchronize_single_particle(int p);
        void get_grid_coordinates(double *x, int *xg, double *xx);
        void linear_interpolation(float *field, int *xg, double *xx, double *dest);
        void rFFTW_to_buffered(rnumber *src, rnumber *dst);

        /* generic methods, should work for all children of this class */
        ptrdiff_t buffered_local_size();
        void read();
        void write();

        /* solvers */
        void Euler();
};


#endif//SLAB_FIELD_PARTICLES

