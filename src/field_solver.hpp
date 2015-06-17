/***********************************************************************
*
*  Copyright 2015 Johns Hopkins University
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
* Contact: turbulence@pha.jhu.edu
* Website: http://turbulence.pha.jhu.edu/
*
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"

#ifndef FLUID_SOLVER

#define FLUID_SOLVER

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * using the same big macro idea that they're using in fftw3.h
 * I feel like I should quote:  Ugh.
 * */

# define FLUID_SOLVER_PROTOTYPES(X, R, C) \
class X(fluid_solver) \
{ \
    public: \
        field_descriptor<R> *fc, *fr; \
 \
        R *rvorticity; \
        R *rvelocity ; \
        C *cvorticity; \
        C *cvelocity ; \
        bool fields_allocated; \
 \
        X(plan) c2r_vorticity; \
        X(plan) r2c_vorticity; \
        X(plan) c2r_velocity; \
        X(plan) r2c_velocity; \
 \
 \
        X(fluid_solver) (); \
        ~X(fluid_solver) (); \
}; \

FLUID_SOLVER_PROTOTYPES(FFTW_MANGLE_DOUBLE, double, fftw_complex)
FLUID_SOLVER_PROTOTYPES(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FLUID_SOLVER_PROTOTYPES(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)

#endif//FLUID_SOLVER

