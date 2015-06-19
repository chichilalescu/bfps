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

#ifndef FLUID_SOLVER

#define FLUID_SOLVER

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * using the same big macro idea that they're using in fftw3.h
 * I feel like I should quote:  Ugh.
 * */

template <class rnumber>
class fluid_solver
{
    private:
        typedef rnumber cnumber[2];
    public:
        field_descriptor<rnumber> *cd, *rd;

        /* fields */
        rnumber *rvorticity;
        rnumber *rvelocity ;
        cnumber *cvorticity;
        cnumber *cvelocity ;

        /* short names for velocity, and 4 vorticity fields */
        rnumber *ru, *rv[4];
        cnumber *cu, *cv[4];

        /* plans */
        void *c2r_vorticity;
        void *r2c_vorticity;
        void *c2r_velocity;
        void *r2c_velocity;
        void *uc2r, *ur2c;
        void *vr2c[3], *vc2r[3];

        /* simulation parameters */
        int iteration;

        /* physical parameters */
        rnumber nu;
        rnumber dkx, dky, dkz, dk;

        /* mode and dealiasing information */
        double kMx, kMy, kMz, kM, kM2;
        double *kx, *ky, *kz;
        bool *knullx, *knully, *knullz;
        int nonzerokx, nonzeroky, nonzerokz;
        double *kshell;
        int64_t *nshell;
        int nshells;


        /* methods */
        fluid_solver(
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0);



        ~fluid_solver();

        void omega_nonlin(int src);
        void step(double dt);
};

#endif//FLUID_SOLVER

