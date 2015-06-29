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
#include "fluid_solver.hpp"
#include "fftw_tools.hpp"


template <class rnumber>
void fluid_solver<rnumber>::impose_zero_modes()
{
    this->cu[0][0] = 0.0;
    this->cu[1][0] = 0.0;
    this->cu[2][0] = 0.0;

    this->cv[0][0][1] = 0.0;
    this->cv[0][1][1] = 0.0;
    this->cv[0][2][1] = 0.0;

    this->cv[1][0][1] = 0.0;
    this->cv[1][1][1] = 0.0;
    this->cv[1][2][1] = 0.0;

    this->cv[2][0][1] = 0.0;
    this->cv[2][1][1] = 0.0;
    this->cv[2][2][1] = 0.0;
}
/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_DEFINITIONS(FFTW, R, C, MPI_RNUM, MPI_CNUM) \
 \
template<> \
fluid_solver<R>::fluid_solver( \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ) : fluid_solver_base<R>(nx , ny , nz, \
                                           DKX, DKY, DKZ) \
{ \
    this->cvorticity = FFTW(alloc_complex)(this->cd->local_size);\
    this->cvelocity  = FFTW(alloc_complex)(this->cd->local_size);\
    this->rvorticity = FFTW(alloc_real)(this->cd->local_size*2);\
    this->rvelocity  = FFTW(alloc_real)(this->cd->local_size*2);\
 \
    this->ru = this->rvelocity;\
    this->cu = this->cvelocity;\
 \
    this->rv[0] = this->rvorticity;\
    this->rv[3] = this->rvorticity;\
    this->cv[0] = this->cvorticity;\
    this->cv[3] = this->cvorticity;\
 \
    this->cv[1] = FFTW(alloc_complex)(this->cd->local_size);\
    this->cv[2] = FFTW(alloc_complex)(this->cd->local_size);\
    this->rv[1] = FFTW(alloc_real)(this->cd->local_size*2);\
    this->rv[2] = FFTW(alloc_real)(this->cd->local_size*2);\
 \
    this->c2r_vorticity = new FFTW(plan);\
    this->r2c_vorticity = new FFTW(plan);\
    this->c2r_velocity  = new FFTW(plan);\
    this->r2c_velocity  = new FFTW(plan);\
 \
    ptrdiff_t sizes[] = {nz, \
                         ny, \
                         nx};\
 \
    *(FFTW(plan)*)this->c2r_vorticity = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->cvorticity, this->rvorticity, \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN); \
 \
    *(FFTW(plan)*)this->r2c_vorticity = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->rvorticity, this->cvorticity, \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT); \
 \
    *(FFTW(plan)*)this->c2r_velocity = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->cvelocity, this->rvelocity, \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN); \
 \
    *(FFTW(plan)*)this->r2c_velocity = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->rvelocity, this->cvelocity, \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT); \
 \
    this->uc2r = this->c2r_velocity;\
    this->ur2c = this->r2c_velocity;\
    this->vc2r[0] = this->c2r_vorticity;\
    this->vr2c[0] = this->r2c_vorticity;\
 \
    this->vc2r[1] = new FFTW(plan);\
    this->vr2c[1] = new FFTW(plan);\
    this->vc2r[2] = new FFTW(plan);\
    this->vr2c[2] = new FFTW(plan);\
 \
    *(FFTW(plan)*)(this->vc2r[1]) = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->cv[1], this->rv[1], \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN); \
 \
    *(FFTW(plan)*)this->vc2r[2] = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->cv[2], this->rv[2], \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN); \
 \
    *(FFTW(plan)*)this->vr2c[1] = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->rv[1], this->cv[1], \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT); \
 \
    *(FFTW(plan)*)this->vr2c[2] = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, \
            this->rv[2], this->cv[2], \
            MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT); \
 \
    /* ``physical'' parameters etc */ \
 \
    this->nu = 0.1; \
} \
 \
template<> \
fluid_solver<R>::~fluid_solver() \
{ \
    FFTW(destroy_plan)(*(FFTW(plan)*)this->c2r_vorticity);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->r2c_vorticity);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->c2r_velocity );\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->r2c_velocity );\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->vc2r[1]);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->vr2c[1]);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->vc2r[2]);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->vr2c[2]);\
 \
    delete (FFTW(plan)*)this->c2r_vorticity;\
    delete (FFTW(plan)*)this->r2c_vorticity;\
    delete (FFTW(plan)*)this->c2r_velocity ;\
    delete (FFTW(plan)*)this->r2c_velocity ;\
    delete (FFTW(plan)*)this->vc2r[1];\
    delete (FFTW(plan)*)this->vr2c[1];\
    delete (FFTW(plan)*)this->vc2r[2];\
    delete (FFTW(plan)*)this->vr2c[2];\
 \
    FFTW(free)(this->cv[1]);\
    FFTW(free)(this->cv[2]);\
    FFTW(free)(this->rv[1]);\
    FFTW(free)(this->rv[2]);\
    FFTW(free)(this->cvorticity);\
    FFTW(free)(this->rvorticity);\
    FFTW(free)(this->cvelocity);\
    FFTW(free)(this->rvelocity);\
} \
 \
template<> \
void fluid_solver<R>::compute_vorticity() \
{ \
    this->low_pass_Fourier(this->cu, 3, this->kM); \
    CLOOP( \
            this->cvorticity[cindex*3+0][0] = (this->ky[yindex]*this->cu[cindex*3+2][1] - this->kz[zindex]*this->cu[cindex*3+1][1]); \
            this->cvorticity[cindex*3+1][0] = (this->kz[zindex]*this->cu[cindex*3+0][1] - this->kx[xindex]*this->cu[cindex*3+2][1]); \
            this->cvorticity[cindex*3+2][0] = (this->kx[xindex]*this->cu[cindex*3+1][1] - this->ky[yindex]*this->cu[cindex*3+0][1]); \
            this->cvorticity[cindex*3+0][1] = (this->ky[yindex]*this->cu[cindex*3+2][0] - this->kz[zindex]*this->cu[cindex*3+1][0]); \
            this->cvorticity[cindex*3+1][1] = (this->kz[zindex]*this->cu[cindex*3+0][0] - this->kx[xindex]*this->cu[cindex*3+2][0]); \
            this->cvorticity[cindex*3+2][1] = (this->kx[xindex]*this->cu[cindex*3+1][0] - this->ky[yindex]*this->cu[cindex*3+0][0]); \
            ); \
    this->impose_zero_modes(); \
    this->symmetrize(this->cvorticity, 3); \
} \
 \
template<> \
void fluid_solver<R>::compute_velocity(C *vorticity) \
{ \
    double k2; \
    this->low_pass_Fourier(vorticity, 3, this->kM); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            this->cu[cindex*3+0][0] = -(this->ky[yindex]*vorticity[cindex*3+2][1] - this->kz[zindex]*vorticity[cindex*3+1][1]) / k2; \
            this->cu[cindex*3+1][0] = -(this->kz[zindex]*vorticity[cindex*3+0][1] - this->kx[xindex]*vorticity[cindex*3+2][1]) / k2; \
            this->cu[cindex*3+2][0] = -(this->kx[xindex]*vorticity[cindex*3+1][1] - this->ky[yindex]*vorticity[cindex*3+0][1]) / k2; \
            this->cu[cindex*3+0][1] =  (this->ky[yindex]*vorticity[cindex*3+2][0] - this->kz[zindex]*vorticity[cindex*3+1][0]) / k2; \
            this->cu[cindex*3+1][1] =  (this->kz[zindex]*vorticity[cindex*3+0][0] - this->kx[xindex]*vorticity[cindex*3+2][0]) / k2; \
            this->cu[cindex*3+2][1] =  (this->kx[xindex]*vorticity[cindex*3+1][0] - this->ky[yindex]*vorticity[cindex*3+0][0]) / k2; \
            ); \
    this->impose_zero_modes(); \
    this->symmetrize(this->cu, 3); \
} \
 \
template<> \
void fluid_solver<R>::omega_nonlin( \
        int src) \
{ \
    assert(src >= 0 && src < 3); \
    /* compute velocity */ \
    this->compute_velocity(this->cv[src]); \
    /* get fields from Fourier space to real space */ \
    FFTW(execute)(*((FFTW(plan)*)this->c2r_velocity )); \
    FFTW(execute)(*((FFTW(plan)*)this->vc2r[src])); \
    /* compute cross product $u \times \omega$, and normalize */ \
    R tmpx0, tmpy0, tmpz0; \
    RLOOP ( \
             tmpx0 = (this->ru[rindex*3+1]*this->rv[src][rindex*3+2] - this->ru[rindex*3+2]*this->rv[src][rindex*3+1]) / this->normalization_factor; \
             tmpy0 = (this->ru[rindex*3+2]*this->rv[src][rindex*3+0] - this->ru[rindex*3+0]*this->rv[src][rindex*3+2]) / this->normalization_factor; \
             tmpz0 = (this->ru[rindex*3+0]*this->rv[src][rindex*3+1] - this->ru[rindex*3+1]*this->rv[src][rindex*3+0]) / this->normalization_factor; \
             this->ru[rindex*3+0] = tmpx0 / this->normalization_factor; \
             this->ru[rindex*3+1] = tmpy0 / this->normalization_factor; \
             this->ru[rindex*3+2] = tmpz0 / this->normalization_factor; \
            ); \
    /* go back to Fourier space */ \
    /* TODO: is 0 padding needed here? */ \
    FFTW(execute)(*((FFTW(plan)*)this->r2c_velocity )); \
    this->low_pass_Fourier(this->cu, 3, this->kM); \
    this->symmetrize(this->cu, 3); \
    /* $\imath k \times DFT(u \times \omega)$ */ \
    R tmpx1, tmpy1, tmpz1; \
/*    FFTW(execute)(*((FFTW(plan)*)this->vr2c[src]));*/ \
    CLOOP( \
            tmpx0 = -(this->ky[yindex]*this->cu[cindex*3+2][1] - this->kz[zindex]*this->cu[cindex*3+1][1]); \
            tmpy0 = -(this->kz[zindex]*this->cu[cindex*3+0][1] - this->kx[xindex]*this->cu[cindex*3+2][1]); \
            tmpz0 = -(this->kx[xindex]*this->cu[cindex*3+1][1] - this->ky[yindex]*this->cu[cindex*3+0][1]); \
            tmpx1 =  (this->ky[yindex]*this->cu[cindex*3+2][0] - this->kz[zindex]*this->cu[cindex*3+1][0]); \
            tmpy1 =  (this->kz[zindex]*this->cu[cindex*3+0][0] - this->kx[xindex]*this->cu[cindex*3+2][0]); \
            tmpz1 =  (this->kx[xindex]*this->cu[cindex*3+1][0] - this->ky[yindex]*this->cu[cindex*3+0][0]); \
            this->cu[cindex*3+0][0] = tmpx0 / this->normalization_factor;\
            this->cu[cindex*3+1][0] = tmpy0 / this->normalization_factor;\
            this->cu[cindex*3+2][0] = tmpz0 / this->normalization_factor;\
            this->cu[cindex*3+0][1] = tmpx1 / this->normalization_factor;\
            this->cu[cindex*3+1][1] = tmpy1 / this->normalization_factor;\
            this->cu[cindex*3+2][1] = tmpz1 / this->normalization_factor;\
            /*this->cv[src][cindex*3+0][0] /= this->normalization_factor;*/ \
            /*this->cv[src][cindex*3+0][1] /= this->normalization_factor;*/ \
            /*this->cv[src][cindex*3+1][0] /= this->normalization_factor;*/ \
            /*this->cv[src][cindex*3+1][1] /= this->normalization_factor;*/ \
            /*this->cv[src][cindex*3+2][0] /= this->normalization_factor;*/ \
            /*this->cv[src][cindex*3+2][1] /= this->normalization_factor;*/ \
            ); \
    this->symmetrize(this->cu, 3); \
    this->symmetrize(this->cv[src], 3); \
} \
 \
template<> \
void fluid_solver<R>::step(double dt) \
{ \
    double k2, factor0, factor1; \
    this->omega_nonlin(0); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            factor0 = exp(-this->nu * k2 * dt); \
            this->cv[1][cindex*3+0][0] = (this->cv[0][cindex*3+0][0] + dt*this->cu[cindex*3+0][0])*factor0; \
            this->cv[1][cindex*3+1][0] = (this->cv[0][cindex*3+1][0] + dt*this->cu[cindex*3+1][0])*factor0; \
            this->cv[1][cindex*3+2][0] = (this->cv[0][cindex*3+2][0] + dt*this->cu[cindex*3+2][0])*factor0; \
            this->cv[1][cindex*3+0][1] = (this->cv[0][cindex*3+0][1] + dt*this->cu[cindex*3+0][1])*factor0; \
            this->cv[1][cindex*3+1][1] = (this->cv[0][cindex*3+1][1] + dt*this->cu[cindex*3+1][1])*factor0; \
            this->cv[1][cindex*3+2][1] = (this->cv[0][cindex*3+2][1] + dt*this->cu[cindex*3+2][1])*factor0; \
            ); \
 \
    this->force_divfree(this->cv[1]); \
    this->omega_nonlin(1); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            factor0 = exp(-this->nu * k2 * dt/2); \
            factor1 = exp( this->nu * k2 * dt/2); \
            if (factor0 > 1 || factor1 > 1) \
            this->cv[2][cindex*3+0][0] = (3*this->cv[0][cindex*3+0][0]*factor0 + (this->cv[1][cindex*3+0][0] + dt*this->cu[cindex*3+0][0])*factor1)*0.25; \
            this->cv[2][cindex*3+1][0] = (3*this->cv[0][cindex*3+1][0]*factor0 + (this->cv[1][cindex*3+1][0] + dt*this->cu[cindex*3+1][0])*factor1)*0.25; \
            this->cv[2][cindex*3+2][0] = (3*this->cv[0][cindex*3+2][0]*factor0 + (this->cv[1][cindex*3+2][0] + dt*this->cu[cindex*3+2][0])*factor1)*0.25; \
            this->cv[2][cindex*3+0][1] = (3*this->cv[0][cindex*3+0][1]*factor0 + (this->cv[1][cindex*3+0][1] + dt*this->cu[cindex*3+0][1])*factor1)*0.25; \
            this->cv[2][cindex*3+1][1] = (3*this->cv[0][cindex*3+1][1]*factor0 + (this->cv[1][cindex*3+1][1] + dt*this->cu[cindex*3+1][1])*factor1)*0.25; \
            this->cv[2][cindex*3+2][1] = (3*this->cv[0][cindex*3+2][1]*factor0 + (this->cv[1][cindex*3+2][1] + dt*this->cu[cindex*3+2][1])*factor1)*0.25; \
            ); \
 \
    this->force_divfree(this->cv[2]); \
    this->omega_nonlin(2); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            factor0 = exp(-this->nu * k2 * dt); \
            factor1 = exp(-this->nu * k2 * dt/2); \
            this->cv[3][cindex*3+0][0] = (this->cv[0][cindex*3+0][0]*factor0 + 2*(this->cv[2][cindex*3+0][0] + dt*this->cu[cindex*3+0][0])*factor1)/3; \
            this->cv[3][cindex*3+1][0] = (this->cv[0][cindex*3+1][0]*factor0 + 2*(this->cv[2][cindex*3+1][0] + dt*this->cu[cindex*3+1][0])*factor1)/3; \
            this->cv[3][cindex*3+2][0] = (this->cv[0][cindex*3+2][0]*factor0 + 2*(this->cv[2][cindex*3+2][0] + dt*this->cu[cindex*3+2][0])*factor1)/3; \
            this->cv[3][cindex*3+0][1] = (this->cv[0][cindex*3+0][1]*factor0 + 2*(this->cv[2][cindex*3+0][1] + dt*this->cu[cindex*3+0][1])*factor1)/3; \
            this->cv[3][cindex*3+1][1] = (this->cv[0][cindex*3+1][1]*factor0 + 2*(this->cv[2][cindex*3+1][1] + dt*this->cu[cindex*3+1][1])*factor1)/3; \
            this->cv[3][cindex*3+2][1] = (this->cv[0][cindex*3+2][1]*factor0 + 2*(this->cv[2][cindex*3+2][1] + dt*this->cu[cindex*3+2][1])*factor1)/3; \
            );  \
    this->force_divfree(this->cv[0]); \
 \
    this->iteration++; \
}
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        fftwf_complex,
        MPI_REAL4,
        MPI_COMPLEX8)
FLUID_SOLVER_DEFINITIONS(
        FFTW_MANGLE_DOUBLE,
        double,
        fftw_complex,
        MPI_REAL8,
        MPI_COMPLEX16)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver<float>;
/*****************************************************************************/

