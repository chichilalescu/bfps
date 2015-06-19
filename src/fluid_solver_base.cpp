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
#include "fluid_solver_base.hpp"
#include "fftw_tools.hpp"



/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_BASE_DEFINITIONS(FFTW, R, C, MPI_RNUM, MPI_CNUM) \
 \
template<> \
fluid_solver_base<R>::fluid_solver_base( \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ) \
{ \
    int ntmp[4]; \
    ntmp[0] = nz; \
    ntmp[1] = ny; \
    ntmp[2] = nx; \
    ntmp[3] = 3; \
    this->rd = new field_descriptor<R>( \
            4, ntmp, MPI_RNUM, MPI_COMM_WORLD);\
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
FLUID_SOLVER_BASE_DEFINITIONS(
        FFTW_MANGLE_DOUBLE,
        double,
        fftw_complex,
        MPI_REAL8,
        MPI_COMPLEX16)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver_base<float>;
/*****************************************************************************/

