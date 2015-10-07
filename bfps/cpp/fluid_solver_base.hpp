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
#include <unordered_map>
#include <vector>
#include "base.hpp"
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
        ptrdiff_t normalization_factor;

        /* simulation parameters */
        char name[256];
        int iteration;

        /* physical parameters */
        double dkx, dky, dkz, dk, dk2;

        /* mode and dealiasing information */
        int dealias_type;
        double kMx, kMy, kMz, kM, kM2;
        double *kx, *ky, *kz;
        std::unordered_map<int, double> Fourier_filter;
        double *kshell;
        int64_t *nshell;
        int nshells;


        /* methods */
        fluid_solver_base(
                const char *NAME,
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0,
                int DEALIAS_TYPE = 0);
        ~fluid_solver_base();

        void low_pass_Fourier(cnumber *a, int howmany, double kmax);
        void dealias(cnumber *a, int howmany);
        void force_divfree(cnumber *a);
        void symmetrize(cnumber *a, int howmany);
        void clean_up_real_space(rnumber *a, int howmany);
        void cospectrum(cnumber *a, cnumber *b, double *spec);
        void cospectrum(cnumber *a, cnumber *b, double *spec, const double k2exponent);
        rnumber autocorrel(cnumber *a);
        void compute_rspace_stats(rnumber *a,
                                  double *moments,
                                  ptrdiff_t *hist,
                                  double max_estimate = 1.0,
                                  int nbins = 256);
        void write_spectrum(const char *fname, cnumber *a, const double k2exponent = 0.0);
        void fill_up_filename(const char *base_name, char *full_name);
        int read_base(const char *fname, rnumber *data);
        int read_base(const char *fname, cnumber *data);
        int write_base(const char *fname, rnumber *data);
        int write_base(const char *fname, cnumber *data);
};



/*****************************************************************************/
/* macros for loops                                                          */

/* Fourier space loop */
#define CLOOP(expression) \
 \
{ \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < this->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < this->cd->subsizes[1]; zindex++) \
    for (ptrdiff_t xindex = 0; xindex < this->cd->subsizes[2]; xindex++) \
        { \
            expression; \
            cindex++; \
        } \
}

#define CLOOP_NXMODES(expression) \
 \
{ \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < this->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < this->cd->subsizes[1]; zindex++) \
    { \
        int nxmodes = 1; \
        ptrdiff_t xindex = 0; \
        expression; \
        cindex++; \
        nxmodes = 2; \
    for (xindex = 1; xindex < this->cd->subsizes[2]; xindex++) \
        { \
            expression; \
            cindex++; \
        } \
    } \
}

#define CLOOP_K2(expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < this->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < this->cd->subsizes[1]; zindex++) \
    for (ptrdiff_t xindex = 0; xindex < this->cd->subsizes[2]; xindex++) \
        { \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            expression; \
            cindex++; \
        } \
}

#define CLOOP_K2_NXMODES(expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < this->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < this->cd->subsizes[1]; zindex++) \
    { \
        int nxmodes = 1; \
        ptrdiff_t xindex = 0; \
        k2 = (this->kx[xindex]*this->kx[xindex] + \
              this->ky[yindex]*this->ky[yindex] + \
              this->kz[zindex]*this->kz[zindex]); \
        expression; \
        cindex++; \
        nxmodes = 2; \
    for (xindex = 1; xindex < this->cd->subsizes[2]; xindex++) \
        { \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            expression; \
            cindex++; \
        } \
    } \
}

/* real space loop */
#define RLOOP(obj, expression) \
 \
{ \
    for (int zindex = 0; zindex < obj->rd->subsizes[0]; zindex++) \
    for (int yindex = 0; yindex < obj->rd->subsizes[1]; yindex++) \
    { \
        ptrdiff_t rindex = (zindex * obj->rd->subsizes[1] + yindex)*(obj->rd->subsizes[2]+2); \
    for (int xindex = 0; xindex < obj->rd->subsizes[2]; xindex++) \
        { \
            expression; \
            rindex++; \
        } \
    } \
}

/*****************************************************************************/

#endif//FLUID_SOLVER_BASE

