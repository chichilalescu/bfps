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

        /* is_active is a matrix of shape [nprocs][nparticles], with
         * is_active[r][p] being true if particle p is in the domain
         * of rank r, or in the buffer regions of this domain.
         * */
        bool **is_active;

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
        slab_field_particles(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER,
                const int NPARTICLES,
                const int NCOMPONENTS,
                const int BUFFERSIZE);
        ~slab_field_particles();

        virtual void get_rhs(double *x, double *rhs);
        /* an Euler step is needed to compute an estimate of future positions,
         * which is needed for synchronization.
         * function is virtual since we want children to do different things,
         * depending on the type of particle. this particular function just
         * copies the old state into the new state.
         * */
        virtual void jump_estimate(double *jump_length);
        void synchronize();
        void rFFTW_to_buffered(rnumber *src, rnumber *dst);
        ptrdiff_t buffered_local_size();
};


#endif//SLAB_FIELD_PARTICLES

