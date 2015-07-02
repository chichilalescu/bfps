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
        int nparticles;

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
        int ncomponents;

        /* simulation parameters */
        char name[256];
        int iteration;

        /* physical parameters of field */
        rnumber dx, dy, dz;

        /* methods */
        slab_field_particles(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER);
        ~slab_field_particles();
};


#endif//SLAB_FIELD_PARTICLES

