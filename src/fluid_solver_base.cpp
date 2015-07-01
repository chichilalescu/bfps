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


#include <cassert>
#include <cmath>
#include <cstring>
#include "fluid_solver_base.hpp"
#include "fftw_tools.hpp"



/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_BASE_DEFINITIONS(FFTW, R, C, MPI_RNUM, MPI_CNUM) \
 \
template<> \
fluid_solver_base<R>::fluid_solver_base( \
        const char *NAME, \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ) \
{ \
    strncpy(this->name, NAME, 256); \
    this->name[255] = '\0'; \
    this->iteration = 0; \
 \
    int ntmp[4]; \
    ntmp[0] = nz; \
    ntmp[1] = ny; \
    ntmp[2] = nx; \
    ntmp[3] = 3; \
    this->rd = new field_descriptor<R>( \
            4, ntmp, MPI_RNUM, MPI_COMM_WORLD);\
    this->normalization_factor = (this->rd->full_size/3); \
    ntmp[0] = ny; \
    ntmp[1] = nz; \
    ntmp[2] = nx/2 + 1; \
    this->cd = new field_descriptor<R>( \
            4, ntmp, MPI_CNUM, MPI_COMM_WORLD);\
 \
    this->dkx = DKX; \
    this->dky = DKY; \
    this->dkz = DKZ; \
    this->kx = new double[this->cd->sizes[2]]; \
    this->ky = new double[this->cd->subsizes[0]]; \
    this->kz = new double[this->cd->sizes[1]]; \
    this->knullx = new bool[this->cd->sizes[2]]; \
    this->knully = new bool[this->cd->subsizes[0]]; \
    this->knullz = new bool[this->cd->sizes[1]]; \
    this->nonzerokx = int(this->rd->sizes[2] / 3); \
    this->kMx = this->dkx*(this->nonzerokx-1); \
    this->nonzeroky = int(this->rd->sizes[1] / 3); \
    this->kMy = this->dky*(this->nonzeroky-1); \
    this->nonzeroky = 2*this->nonzeroky - 1; \
    this->nonzerokz = int(this->rd->sizes[0] / 3); \
    this->kMz = this->dkz*(this->nonzerokz-1); \
    this->nonzerokz = 2*this->nonzerokz - 1; \
    int i, ii; \
    for (i = 0; i<this->cd->sizes[2]; i++) \
    { \
        this->kx[i] = i*this->dkx; \
        if (i < this->nonzerokx) \
            this->knullx[i] = false; \
        else \
            this->knullx[i] = true; \
    } \
    for (i = 0; i<this->cd->subsizes[0]; i++) \
    { \
        int tval = (this->nonzeroky+1)/2; \
        ii = i + this->cd->starts[0]; \
        if (ii <= this->rd->sizes[1]/2) \
            this->ky[i] = this->dky*ii; \
        else \
            this->ky[i] = this->dky*(ii - this->rd->sizes[1]); \
        if (ii < tval || (this->rd->sizes[1] - ii) < tval) \
            this->knully[i] = false; \
        else \
            this->knully[i] = true; \
    } \
    for (i = 0; i<this->cd->sizes[1]; i++) \
    { \
        int tval = (this->nonzerokz+1)/2; \
        if (i <= this->rd->sizes[0]/2) \
            this->kz[i] = this->dkz*i; \
        else \
            this->kz[i] = this->dkz*(i - this->rd->sizes[0]); \
        if (i < tval || (this->rd->sizes[0] - i) < tval) \
            this->knullz[i] = false; \
        else \
            this->knullz[i] = true; \
    } \
    this->kM = this->kMx; \
    if (this->kM < this->kMy) this->kM = this->kMy; \
    if (this->kM < this->kMz) this->kM = this->kMz; \
    this->kM2 = this->kM * this->kM; \
    this->dk = this->dkx; \
    if (this->dk > this->dky) this->dk = this->dky; \
    if (this->dk > this->dkz) this->dk = this->dkz; \
    /*for (i = 0; i<this->cd->sizes[2]; i++) \
        DEBUG_MSG("kx[%d] = %lg\n", i, this->kx[i]);*/ \
    /*for (i = 0; i<this->cd->subsizes[0]; i++) \
        DEBUG_MSG("ky[%d] = %lg\n", i, this->ky[i]);*/ \
    /*for (i = 0; i<this->cd->sizes[1]; i++) \
        DEBUG_MSG("kz[%d] = %lg\n", i, this->kz[i]);*/ \
} \
 \
template<> \
fluid_solver_base<R>::~fluid_solver_base() \
{ \
 \
    delete this->kx;\
    delete this->ky;\
    delete this->kz;\
    delete this->knullx;\
    delete this->knully;\
    delete this->knullz;\
 \
    delete this->cd; \
    delete this->rd; \
} \
 \
template<> \
R fluid_solver_base<R>::correl_vec(C *a, C *b) \
{ \
    double val_process = 0.0, val; \
    double k2; \
    int factor = 1;\
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            if (k2 < this->kM2) \
            { \
                factor = (xindex == 0) ? 1 : 2; \
                val_process += factor * ((*(a + 3*cindex))[0] * (*(b + 3*cindex))[0] + \
                                         (*(a + 3*cindex))[1] * (*(b + 3*cindex))[1] + \
                                         (*(a + 3*cindex+1))[0] * (*(b + 3*cindex+1))[0] + \
                                         (*(a + 3*cindex+1))[1] * (*(b + 3*cindex+1))[1] + \
                                         (*(a + 3*cindex+2))[0] * (*(b + 3*cindex+2))[0] + \
                                         (*(a + 3*cindex+2))[1] * (*(b + 3*cindex+2))[1] \
                                        ); \
            } \
            );\
    MPI_Allreduce( \
            (void*)(&val_process), \
            (void*)(&val), \
            1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD); \
    return R(val); \
} \
 \
template<> \
void fluid_solver_base<R>::low_pass_Fourier(C *a, const int howmany, const double kmax) \
{ \
    double k2; \
    const double km2 = kmax*kmax; \
    const int howmany2 = 2*howmany; \
    /*DEBUG_MSG("entered low_pass_Fourier, kmax=%lg km2=%lg howmany2=%d\n", kmax, km2, howmany2);*/ \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            /*DEBUG_MSG("kx=%lg ky=%lg kz=%lg k2=%lg\n", \
                      this->kx[xindex], \
                      this->ky[yindex], \
                      this->kz[zindex], \
                      k2);*/ \
            if (k2 >= km2) \
            { \
                for (int tcounter = 0; tcounter < howmany; tcounter++) \
                { \
                    a[howmany*cindex+tcounter][0] = 0.0; \
                    a[howmany*cindex+tcounter][1] = 0.0; \
                } \
                /*std::fill_n((R*)(a + howmany*cindex), howmany2, 0.0);*/ \
            } \
            );\
} \
 \
template<> \
void fluid_solver_base<R>::force_divfree(C *a) \
{ \
    double k2; \
    C tval; \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            if (k2 >= this->kM2) \
            { \
                tval[0] = (this->kx[xindex]*((*(a + cindex*3  ))[0]) + \
                           this->ky[yindex]*((*(a + cindex*3+1))[0]) + \
                           this->kz[zindex]*((*(a + cindex*3+2))[0]) ) / k2; \
                tval[1] = (this->kx[xindex]*((*(a + cindex*3  ))[1]) + \
                           this->ky[yindex]*((*(a + cindex*3+1))[1]) + \
                           this->kz[zindex]*((*(a + cindex*3+2))[1]) ) / k2; \
                a[cindex*3  ][0] -= tval[0]*this->kx[xindex]; \
                a[cindex*3+1][1] -= tval[1]*this->kx[xindex]; \
                a[cindex*3+2][0] -= tval[0]*this->ky[yindex]; \
                a[cindex*3  ][1] -= tval[1]*this->ky[yindex]; \
                a[cindex*3+1][0] -= tval[0]*this->kz[zindex]; \
                a[cindex*3+2][1] -= tval[1]*this->kz[zindex]; \
            } \
            );\
} \
 \
template<> \
void fluid_solver_base<R>::symmetrize(C *data, const int howmany) \
{ \
    ptrdiff_t ii, cc; \
    MPI_Status *mpistatus = new MPI_Status[1]; \
    if (this->cd->myrank == this->cd->rank[0]) \
    { \
        for (cc = 0; cc < howmany; cc++) \
            data[cc][1] = 0.0; \
        for (ii = 1; ii < this->cd->sizes[1]/2; ii++) \
            for (cc = 0; cc < howmany; cc++) { \
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[0] = \
                 (*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[0]; \
                ( *(data + cc + howmany*(this->cd->sizes[1] - ii)*this->cd->sizes[2]))[1] = \
                -(*(data + cc + howmany*(                     ii)*this->cd->sizes[2]))[1]; \
                } \
    } \
    C *buffer; \
    buffer = FFTW(alloc_complex)(howmany*this->cd->sizes[1]); \
    ptrdiff_t yy; \
    int ranksrc, rankdst; \
    for (yy = 1; yy < this->cd->sizes[0]/2; yy++) { \
        ranksrc = this->cd->rank[yy]; \
        rankdst = this->cd->rank[this->cd->sizes[0] - yy]; \
        if (this->cd->myrank == ranksrc) \
            for (ii = 0; ii < this->cd->sizes[1]; ii++) \
                for (cc = 0; cc < howmany; cc++) { \
                    (*(buffer + howmany*ii+cc))[0] = \
                        (*((data + howmany*((yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2]) + cc))[0]; \
                    (*(buffer + howmany*ii+cc))[1] = \
                        (*((data + howmany*((yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2]) + cc))[1]; \
                } \
        if (ranksrc != rankdst) \
        { \
            if (this->cd->myrank == ranksrc) \
                MPI_Send((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, rankdst, yy, \
                         MPI_COMM_WORLD); \
            if (this->cd->myrank == rankdst) \
                MPI_Recv((void*)buffer, \
                         howmany*this->cd->sizes[1], MPI_CNUM, ranksrc, yy, \
                         MPI_COMM_WORLD, mpistatus); \
        } \
        if (this->cd->myrank == rankdst) \
        { \
            for (ii = 1; ii < this->cd->sizes[1]; ii++) \
                for (cc = 0; cc < howmany; cc++) { \
                    (*((data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2]) + cc))[0] = \
                        (*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[0]; \
                    (*((data + howmany*((this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1] + ii)*this->cd->sizes[2]) + cc))[1] = \
                       -(*(buffer + howmany*(this->cd->sizes[1]-ii)+cc))[1]; \
                } \
            for (cc = 0; cc < howmany; cc++) { \
                (*((data + (this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2] + cc)))[0] =  (*(buffer + cc))[0]; \
                (*((data + (this->cd->sizes[0] - yy - this->cd->starts[0])*this->cd->sizes[1]*this->cd->sizes[2] + cc)))[1] = -(*(buffer + cc))[1]; \
            } \
        } \
    } \
    FFTW(free)(buffer); \
    delete mpistatus; \
    /* put asymmetric data to 0 */\
    if (this->cd->myrank == this->cd->rank[this->cd->sizes[0]/2]) \
    { \
        data[(this->cd->sizes[0]/2 - this->cd->starts[0])*this->sizes[1]] \
    } \
} \
 \
template<> \
int fluid_solver_base<R>::read_base(const char *fname, R *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->rd->read(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::read_base(const char *fname, C *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->cd->read(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::write_base(const char *fname, R *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->rd->write(full_name, (void*)data); \
} \
 \
template<> \
int fluid_solver_base<R>::write_base(const char *fname, C *data) \
{ \
    char full_name[512]; \
    sprintf(full_name, "%s_%s_i%.5x", this->name, fname, this->iteration); \
    return this->cd->write(full_name, (void*)data); \
}

/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_BASE_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        fftwf_complex,
        MPI_REAL4,
        MPI_COMPLEX8)
//FLUID_SOLVER_BASE_DEFINITIONS(
//        FFTW_MANGLE_DOUBLE,
//        double,
//        fftw_complex,
//        MPI_REAL8,
//        MPI_COMPLEX16)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver_base<float>;
/*****************************************************************************/

