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

#ifndef FLUID_SOLVER_BASE

#define FLUID_SOLVER_BASE

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * using the same big macro idea that they're using in fftw3.h
 * I feel like I should quote:  Ugh.
 * */

template <class rnumber>
class fluid_solver_base
{
    protected:
        typedef rnumber cnumber[2];
    public:
        field_descriptor<rnumber> *cd, *rd;

        /* simulation parameters */
        int iteration;

        /* physical parameters */
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
        fluid_solver_base(
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0);
        ~fluid_solver_base();
};



/*****************************************************************************/
/* macros for loops                                                          */

/* Fourier space loop */
#define CLOOP(expression) \
 \
{ \
    int cindex; \
    for (int yindex = 0; yindex < this->cd->subsizes[0]; yindex++) \
    for (int zindex = 0; zindex < this->cd->subsizes[1]; zindex++) \
    { \
        cindex = (yindex * this->cd->sizes[1] + zindex)*this->cd->sizes[2]; \
    for (int xindex = 0; xindex < this->cd->subsizes[2]; xindex++) \
        { \
            expression; \
            cindex++; \
        } \
    } \
}

/* real space loop */
#define RLOOP(expression) \
 \
{ \
    int rindex; \
    for (int zindex = 0; zindex < this->rd->subsizes[0]; zindex++) \
    for (int yindex = 0; yindex < this->rd->subsizes[1]; yindex++) \
    { \
        rindex = (zindex * this->rd->sizes[1] + yindex)*this->rd->sizes[2]; \
    for (int xindex = 0; xindex < this->rd->subsizes[2]; xindex++) \
        { \
            expression; \
            rindex++; \
        } \
    } \
}

/*****************************************************************************/

#endif//FLUID_SOLVER_BASE

