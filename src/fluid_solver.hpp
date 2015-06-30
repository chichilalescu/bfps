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
#include "field_descriptor.hpp"
#include "vector_field.hpp"
#include "fluid_solver_base.hpp"

#ifndef FLUID_SOLVER

#define FLUID_SOLVER

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * using the same big macro idea that they're using in fftw3.h
 * I feel like I should quote:  Ugh.
 * */

template <class rnumber>
class fluid_solver:public fluid_solver_base<rnumber>
{
    public:
        /* fields */
        rnumber *rvorticity;
        rnumber *rvelocity ;
        typename fluid_solver_base<rnumber>::cnumber *cvorticity;
        typename fluid_solver_base<rnumber>::cnumber *cvelocity ;

        /* short names for velocity, and 4 vorticity fields */
        rnumber *ru, *rv[4];
        typename fluid_solver_base<rnumber>::cnumber *cu, *cv[4];

        /* plans */
        void *c2r_vorticity;
        void *r2c_vorticity;
        void *c2r_velocity;
        void *r2c_velocity;
        void *uc2r, *ur2c;
        void *vr2c[3], *vc2r[3];

        /* physical parameters */
        double nu;
        int fmode;
        double famplitude;

        /* methods */
        fluid_solver(
                const char *NAME,
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0);
        ~fluid_solver(void);

        void compute_vorticity(void);
        void compute_velocity(rnumber (*vorticity)[2]);
        void omega_nonlin(int src);
        void step(double dt);
        void impose_zero_modes(void);
        void add_forcing(rnumber (*field)[2], rnumber factor);

        int read(char field, char representation);
        int write(char field, char representation);
};

#endif//FLUID_SOLVER

