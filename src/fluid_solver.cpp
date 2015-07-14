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
    this->cu[0][1] = 0.0;
    this->cu[1][1] = 0.0;
    this->cu[2][1] = 0.0;

    this->cv[0][0][0] = 0.0;
    this->cv[0][1][0] = 0.0;
    this->cv[0][2][0] = 0.0;

    this->cv[1][0][0] = 0.0;
    this->cv[1][1][0] = 0.0;
    this->cv[1][2][0] = 0.0;

    this->cv[2][0][0] = 0.0;
    this->cv[2][1][0] = 0.0;
    this->cv[2][2][0] = 0.0;

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
        const char *NAME, \
        int nx, \
        int ny, \
        int nz, \
        double DKX, \
        double DKY, \
        double DKZ) : fluid_solver_base<R>(NAME, \
                                           nx , ny , nz, \
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
            this->cvorticity[3*cindex+0][0] = (this->ky[yindex]*this->cu[3*cindex+2][1] - this->kz[zindex]*this->cu[3*cindex+1][1]); \
            this->cvorticity[3*cindex+1][0] = (this->kz[zindex]*this->cu[3*cindex+0][1] - this->kx[xindex]*this->cu[3*cindex+2][1]); \
            this->cvorticity[3*cindex+2][0] = (this->kx[xindex]*this->cu[3*cindex+1][1] - this->ky[yindex]*this->cu[3*cindex+0][1]); \
            this->cvorticity[3*cindex+0][1] = (this->ky[yindex]*this->cu[3*cindex+2][0] - this->kz[zindex]*this->cu[3*cindex+1][0]); \
            this->cvorticity[3*cindex+1][1] = (this->kz[zindex]*this->cu[3*cindex+0][0] - this->kx[xindex]*this->cu[3*cindex+2][0]); \
            this->cvorticity[3*cindex+2][1] = (this->kx[xindex]*this->cu[3*cindex+1][0] - this->ky[yindex]*this->cu[3*cindex+0][0]); \
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
    std::fill_n((R*)this->cu, this->cd->local_size*2, 0.0); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            if (k2 < this->kM2) \
            { \
                this->cu[3*cindex+0][0] = -(this->ky[yindex]*vorticity[3*cindex+2][1] - this->kz[zindex]*vorticity[3*cindex+1][1]) / k2; \
                this->cu[3*cindex+1][0] = -(this->kz[zindex]*vorticity[3*cindex+0][1] - this->kx[xindex]*vorticity[3*cindex+2][1]) / k2; \
                this->cu[3*cindex+2][0] = -(this->kx[xindex]*vorticity[3*cindex+1][1] - this->ky[yindex]*vorticity[3*cindex+0][1]) / k2; \
                this->cu[3*cindex+0][1] =  (this->ky[yindex]*vorticity[3*cindex+2][0] - this->kz[zindex]*vorticity[3*cindex+1][0]) / k2; \
                this->cu[3*cindex+1][1] =  (this->kz[zindex]*vorticity[3*cindex+0][0] - this->kx[xindex]*vorticity[3*cindex+2][0]) / k2; \
                this->cu[3*cindex+2][1] =  (this->kx[xindex]*vorticity[3*cindex+1][0] - this->ky[yindex]*vorticity[3*cindex+0][0]) / k2; \
            } \
            else \
            { \
                this->cu[3*cindex+0][0] = 0.0; \
                this->cu[3*cindex+1][0] = 0.0; \
                this->cu[3*cindex+2][0] = 0.0; \
                this->cu[3*cindex+0][1] = 0.0; \
                this->cu[3*cindex+1][1] = 0.0; \
                this->cu[3*cindex+2][1] = 0.0; \
            } \
            ); \
    this->impose_zero_modes(); \
    this->symmetrize(this->cu, 3); \
    /*this->low_pass_Fourier(this->cu, 3, this->kM);*/ \
} \
 \
template<> \
void fluid_solver<R>::ift_velocity() \
{ \
    FFTW(execute)(*((FFTW(plan)*)this->c2r_velocity )); \
} \
 \
template<> \
void fluid_solver<R>::add_forcing(\
        C *field, R factor) \
{ \
    ptrdiff_t cindex; \
    if (this->cd->myrank == this->cd->rank[this->fmode]) \
    { \
        cindex = ((this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3; \
        field[cindex+2][0] -= this->famplitude*factor/2; \
    } \
    if (this->cd->myrank == this->cd->rank[this->cd->sizes[0] - this->fmode]) \
    { \
        cindex = ((this->cd->sizes[0] - this->fmode - this->cd->starts[0]) * this->cd->sizes[1])*this->cd->sizes[2]*3; \
        field[cindex+2][0] -= this->famplitude*factor/2; \
    } \
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
    /*std::fill_n(this->ru, 2*this->cd->local_size, 0);     */ \
    /*std::fill_n(this->rv[src], 2*this->cd->local_size, 0);*/ \
    FFTW(execute)(*((FFTW(plan)*)this->c2r_velocity ));  \
    FFTW(execute)(*((FFTW(plan)*)this->vc2r[src]));      \
    /* compute cross product $u \times \omega$, and normalize */ \
    R tmpx0, tmpy0, tmpz0; \
    RLOOP ( \
             tmpx0 = (this->ru[(3*rindex)+1]*this->rv[src][(3*rindex)+2] - this->ru[(3*rindex)+2]*this->rv[src][(3*rindex)+1]); \
             tmpy0 = (this->ru[(3*rindex)+2]*this->rv[src][(3*rindex)+0] - this->ru[(3*rindex)+0]*this->rv[src][(3*rindex)+2]); \
             tmpz0 = (this->ru[(3*rindex)+0]*this->rv[src][(3*rindex)+1] - this->ru[(3*rindex)+1]*this->rv[src][(3*rindex)+0]); \
             this->ru[(3*rindex)+0] = tmpx0 / this->normalization_factor; \
             this->ru[(3*rindex)+1] = tmpy0 / this->normalization_factor; \
             this->ru[(3*rindex)+2] = tmpz0 / this->normalization_factor; \
            ); \
    /* go back to Fourier space */ \
    FFTW(execute)(*((FFTW(plan)*)this->r2c_velocity )); \
    this->low_pass_Fourier(this->cu, 3, this->kM); \
    this->symmetrize(this->cu, 3); \
    /* $\imath k \times Fourier(u \times \omega)$ */ \
    R tmpx1, tmpy1, tmpz1; \
    CLOOP( \
            tmpx0 = -(this->ky[yindex]*this->cu[3*cindex+2][1] - this->kz[zindex]*this->cu[3*cindex+1][1]); \
            tmpy0 = -(this->kz[zindex]*this->cu[3*cindex+0][1] - this->kx[xindex]*this->cu[3*cindex+2][1]); \
            tmpz0 = -(this->kx[xindex]*this->cu[3*cindex+1][1] - this->ky[yindex]*this->cu[3*cindex+0][1]); \
            tmpx1 =  (this->ky[yindex]*this->cu[3*cindex+2][0] - this->kz[zindex]*this->cu[3*cindex+1][0]); \
            tmpy1 =  (this->kz[zindex]*this->cu[3*cindex+0][0] - this->kx[xindex]*this->cu[3*cindex+2][0]); \
            tmpz1 =  (this->kx[xindex]*this->cu[3*cindex+1][0] - this->ky[yindex]*this->cu[3*cindex+0][0]); \
            this->cu[3*cindex+0][0] = tmpx0; \
            this->cu[3*cindex+1][0] = tmpy0; \
            this->cu[3*cindex+2][0] = tmpz0; \
            this->cu[3*cindex+0][1] = tmpx1; \
            this->cu[3*cindex+1][1] = tmpy1; \
            this->cu[3*cindex+2][1] = tmpz1; \
            ); \
    /*this->symmetrize(this->cu, 3);*/ \
    this->add_forcing(this->cu, 1.0); \
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
            this->cv[1][3*cindex+0][0] = (this->cv[0][3*cindex+0][0] + dt*this->cu[3*cindex+0][0])*factor0; \
            this->cv[1][3*cindex+1][0] = (this->cv[0][3*cindex+1][0] + dt*this->cu[3*cindex+1][0])*factor0; \
            this->cv[1][3*cindex+2][0] = (this->cv[0][3*cindex+2][0] + dt*this->cu[3*cindex+2][0])*factor0; \
            this->cv[1][3*cindex+0][1] = (this->cv[0][3*cindex+0][1] + dt*this->cu[3*cindex+0][1])*factor0; \
            this->cv[1][3*cindex+1][1] = (this->cv[0][3*cindex+1][1] + dt*this->cu[3*cindex+1][1])*factor0; \
            this->cv[1][3*cindex+2][1] = (this->cv[0][3*cindex+2][1] + dt*this->cu[3*cindex+2][1])*factor0; \
            ); \
 \
    this->low_pass_Fourier(this->cv[1], 3, this->kM); \
    this->force_divfree(this->cv[1]); \
    this->omega_nonlin(1); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            factor0 = exp(-this->nu * k2 * dt/2); \
            factor1 = exp( this->nu * k2 * dt/2); \
            this->cv[2][3*cindex+0][0] = (3*this->cv[0][3*cindex+0][0]*factor0 + (this->cv[1][3*cindex+0][0] + dt*this->cu[3*cindex+0][0])*factor1)*0.25; \
            this->cv[2][3*cindex+1][0] = (3*this->cv[0][3*cindex+1][0]*factor0 + (this->cv[1][3*cindex+1][0] + dt*this->cu[3*cindex+1][0])*factor1)*0.25; \
            this->cv[2][3*cindex+2][0] = (3*this->cv[0][3*cindex+2][0]*factor0 + (this->cv[1][3*cindex+2][0] + dt*this->cu[3*cindex+2][0])*factor1)*0.25; \
            this->cv[2][3*cindex+0][1] = (3*this->cv[0][3*cindex+0][1]*factor0 + (this->cv[1][3*cindex+0][1] + dt*this->cu[3*cindex+0][1])*factor1)*0.25; \
            this->cv[2][3*cindex+1][1] = (3*this->cv[0][3*cindex+1][1]*factor0 + (this->cv[1][3*cindex+1][1] + dt*this->cu[3*cindex+1][1])*factor1)*0.25; \
            this->cv[2][3*cindex+2][1] = (3*this->cv[0][3*cindex+2][1]*factor0 + (this->cv[1][3*cindex+2][1] + dt*this->cu[3*cindex+2][1])*factor1)*0.25; \
            ); \
 \
    this->low_pass_Fourier(this->cv[2], 3, this->kM); \
    this->force_divfree(this->cv[2]); \
    this->omega_nonlin(2); \
    CLOOP( \
            k2 = (this->kx[xindex]*this->kx[xindex] + \
                  this->ky[yindex]*this->ky[yindex] + \
                  this->kz[zindex]*this->kz[zindex]); \
            factor0 = exp(-this->nu * k2 * dt * 0.5); \
            this->cv[3][3*cindex+0][0] = (this->cv[0][3*cindex+0][0]*factor0 + 2*(this->cv[2][3*cindex+0][0] + dt*this->cu[3*cindex+0][0]))*factor0/3; \
            this->cv[3][3*cindex+1][0] = (this->cv[0][3*cindex+1][0]*factor0 + 2*(this->cv[2][3*cindex+1][0] + dt*this->cu[3*cindex+1][0]))*factor0/3; \
            this->cv[3][3*cindex+2][0] = (this->cv[0][3*cindex+2][0]*factor0 + 2*(this->cv[2][3*cindex+2][0] + dt*this->cu[3*cindex+2][0]))*factor0/3; \
            this->cv[3][3*cindex+0][1] = (this->cv[0][3*cindex+0][1]*factor0 + 2*(this->cv[2][3*cindex+0][1] + dt*this->cu[3*cindex+0][1]))*factor0/3; \
            this->cv[3][3*cindex+1][1] = (this->cv[0][3*cindex+1][1]*factor0 + 2*(this->cv[2][3*cindex+1][1] + dt*this->cu[3*cindex+1][1]))*factor0/3; \
            this->cv[3][3*cindex+2][1] = (this->cv[0][3*cindex+2][1]*factor0 + 2*(this->cv[2][3*cindex+2][1] + dt*this->cu[3*cindex+2][1]))*factor0/3; \
            );  \
    this->low_pass_Fourier(this->cv[0], 3, this->kM); \
    this->force_divfree(this->cv[0]); \
 \
    this->iteration++; \
} \
 \
template<> \
int fluid_solver<R>::read(char field, char representation) \
{ \
    int read_result; \
    if (field == 'v') \
    { \
        if (representation == 'c') \
        { \
            read_result = this->read_base("cvorticity", this->cvorticity); \
            if (read_result != EXIT_SUCCESS) \
                return read_result; \
        } \
        if (representation == 'r') \
        { \
            read_result = this->read_base("rvorticity", this->rvorticity); \
            if (read_result != EXIT_SUCCESS) \
                return read_result; \
            else \
                FFTW(execute)(*((FFTW(plan)*)this->r2c_vorticity )); \
        } \
        this->low_pass_Fourier(this->cvorticity, 3, this->kM); \
        return EXIT_SUCCESS; \
    } \
    if ((field == 'u') && (representation == 'c')) \
        return this->read_base("cvelocity", this->cvelocity); \
    if ((field == 'u') && (representation == 'r')) \
        return this->read_base("rvelocity", this->rvelocity); \
    return EXIT_FAILURE; \
} \
 \
template<> \
int fluid_solver<R>::write(char field, char representation) \
{ \
    if ((field == 'v') && (representation == 'c')) \
        return this->write_base("cvorticity", this->cvorticity); \
    if ((field == 'v') && (representation == 'r')) \
    { \
        FFTW(execute)(*((FFTW(plan)*)this->c2r_vorticity )); \
        clip_zero_padding<R>(this->rd, this->rvorticity, 3); \
        return this->write_base("rvorticity", this->rvorticity); \
    } \
    this->compute_velocity(this->cvorticity); \
    if ((field == 'u') && (representation == 'c')) \
        return this->write_base("cvelocity", this->cvelocity); \
    if ((field == 'u') && (representation == 'r')) \
    { \
        FFTW(execute)(*((FFTW(plan)*)this->c2r_velocity )); \
        clip_zero_padding<R>(this->rd, this->rvelocity, 3); \
        return this->write_base("rvelocity", this->rvelocity); \
    } \
    return EXIT_FAILURE; \
}
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        fftwf_complex,
        MPI_FLOAT,
        MPI_COMPLEX)
//FLUID_SOLVER_DEFINITIONS(
//        FFTW_MANGLE_DOUBLE,
//        double,
//        fftw_complex,
//        MPI_DOUBLE,
//        MPI_C_DOUBLE_COMPLEX)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver<float>;
/*****************************************************************************/

