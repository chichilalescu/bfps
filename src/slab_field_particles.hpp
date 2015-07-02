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
        fluid_solver_base<rnumber> *rd;
        int nparticles;
        bool 

        /* simulation parameters */
        char name[256];
        int iteration;

        /* physical parameters of field */
        rnumber dx, dy, dz;

        /* methods */
        slab_field_particles(
                const char *NAME,
                int nx,
                int ny,
                int nz,
                double DX = 1.0,
                double DY = 1.0,
                double DZ = 1.0);
        ~slab_field_particles();

        void low_pass_Fourier(cnumber *a, int howmany, double kmax);
        void force_divfree(cnumber *a);
        void symmetrize(cnumber *a, int howmany);
        rnumber correl_vec(cnumber *a, cnumber *b);
        int read_base(const char *fname, rnumber *data);
        int read_base(const char *fname, cnumber *data);
        int write_base(const char *fname, rnumber *data);
        int write_base(const char *fname, cnumber *data);
};


#endif//SLAB_FIELD_PARTICLES

