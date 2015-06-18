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
        bool fields_allocated;

        /* plans */
        void *c2r_vorticity;
        void *r2c_vorticity;
        void *c2r_velocity;
        void *r2c_velocity;

        /* simulation parameters */
        int iteration;

        /* physical parameters */
        rnumber nu;


        /* methods */
        fluid_solver(
                int nx, int ny, int nz);
        ~fluid_solver();

        void step();
};

#endif//FLUID_SOLVER

