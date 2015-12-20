/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/

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
        unsigned fftw_plan_rigor;

        /* simulation parameters */
        char name[256];
        int iteration;

        /* physical parameters */
        double dkx, dky, dkz, dk, dk2;

        /* mode and dealiasing information */
        int dealias_type;
        double kMx, kMy, kMz, kM, kM2;
        double kMspec, kMspec2;
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
                int DEALIAS_TYPE = 0,
                unsigned FFTW_PLAN_RIGOR = FFTW_ESTIMATE);
        ~fluid_solver_base();

        void low_pass_Fourier(cnumber *__restrict__ a, int howmany, double kmax);
        void dealias(cnumber *__restrict__ a, int howmany);
        void force_divfree(cnumber *__restrict__ a);
        void symmetrize(cnumber *__restrict__ a, int howmany);
        void clean_up_real_space(rnumber *__restrict__ a, int howmany);
        void cospectrum(cnumber *__restrict__ a, cnumber *__restrict__ b, double *__restrict__ spec);
        void cospectrum(cnumber *__restrict__ a, cnumber *__restrict__ b, double *__restrict__ spec, const double k2exponent);
        double autocorrel(cnumber *__restrict__ a);
        template <int nvals>
        void compute_rspace_stats(rnumber *__restrict__ a,
                                  double *__restrict__ moments,
                                  ptrdiff_t *__restrict__ hist,
                                  double max_estimate[nvals],
                                  const int nbins = 256);
        inline void compute_rspace_stats3(rnumber *__restrict__ a,
                                  double *__restrict__ moments,
                                  ptrdiff_t *__restrict__ hist,
                                  double max_estimate[3],
                                  const int nbins = 256)
        {
            this->compute_rspace_stats<3>(a, moments, hist, max_estimate, nbins);
        }
        inline void compute_rspace_stats4(rnumber *__restrict__ a,
                                  double *__restrict__ moments,
                                  ptrdiff_t *__restrict__ hist,
                                  double max_estimate[4],
                                  const int nbins = 256)
        {
            this->compute_rspace_stats<4>(a, moments, hist, max_estimate, nbins);
        }
        void compute_vector_gradient(rnumber (*__restrict__ A)[2], rnumber(*__restrict__ source)[2]);
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
#define CLOOP(obj, expression) \
 \
{ \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < obj->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < obj->cd->subsizes[1]; zindex++) \
    for (ptrdiff_t xindex = 0; xindex < obj->cd->subsizes[2]; xindex++) \
        { \
            expression; \
            cindex++; \
        } \
}

#define CLOOP_NXMODES(obj, expression) \
 \
{ \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < obj->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < obj->cd->subsizes[1]; zindex++) \
    { \
        int nxmodes = 1; \
        ptrdiff_t xindex = 0; \
        expression; \
        cindex++; \
        nxmodes = 2; \
    for (xindex = 1; xindex < obj->cd->subsizes[2]; xindex++) \
        { \
            expression; \
            cindex++; \
        } \
    } \
}

#define CLOOP_K2(obj, expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < obj->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < obj->cd->subsizes[1]; zindex++) \
    for (ptrdiff_t xindex = 0; xindex < obj->cd->subsizes[2]; xindex++) \
        { \
            k2 = (obj->kx[xindex]*obj->kx[xindex] + \
                  obj->ky[yindex]*obj->ky[yindex] + \
                  obj->kz[zindex]*obj->kz[zindex]); \
            expression; \
            cindex++; \
        } \
}

#define CLOOP_K2_NXMODES(obj, expression) \
 \
{ \
    double k2; \
    ptrdiff_t cindex = 0; \
    for (ptrdiff_t yindex = 0; yindex < obj->cd->subsizes[0]; yindex++) \
    for (ptrdiff_t zindex = 0; zindex < obj->cd->subsizes[1]; zindex++) \
    { \
        int nxmodes = 1; \
        ptrdiff_t xindex = 0; \
        k2 = (obj->kx[xindex]*obj->kx[xindex] + \
              obj->ky[yindex]*obj->ky[yindex] + \
              obj->kz[zindex]*obj->kz[zindex]); \
        expression; \
        cindex++; \
        nxmodes = 2; \
    for (xindex = 1; xindex < obj->cd->subsizes[2]; xindex++) \
        { \
            k2 = (obj->kx[xindex]*obj->kx[xindex] + \
                  obj->ky[yindex]*obj->ky[yindex] + \
                  obj->kz[zindex]*obj->kz[zindex]); \
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

