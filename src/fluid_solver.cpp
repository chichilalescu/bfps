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



#include "fluid_solver.hpp"
#include "fftwf_tools.hpp"



/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_DEFINITIONS(FFTW, R, C) \
 \
template<> \
fluid_solver<R>::fluid_solver( \
        int nx, \
        int ny, \
        int nz) \
{ \
    FFTW(get_descriptors_3D)(nz, ny, nx, &this->fr, &this->fc);\
    this->cvorticity = FFTW(alloc_complex)(this->fc->local_size*3);\
    this->cvelocity  = FFTW(alloc_complex)(this->fc->local_size*3);\
    this->rvorticity = FFTW(alloc_real)(this->fc->local_size*6);\
    this->rvelocity  = FFTW(alloc_real)(this->fc->local_size*6);\
 \
    this->c2r_vorticity = new FFTW(plan);\
    this->r2c_vorticity = new FFTW(plan);\
    this->c2r_velocity  = new FFTW(plan);\
    this->r2c_velocity  = new FFTW(plan);\
 \
    ptrdiff_t sizes[] = {this->fr->sizes[0], \
                         this->fr->sizes[1], \
                         this->fr->sizes[2]};\
 \
    *(FFTW(plan)*)this->c2r_vorticity = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, \
            FFTW_MPI_DEFAULT_BLOCK, \
            FFTW_MPI_DEFAULT_BLOCK, \
            this->cvorticity, this->rvorticity, \
            MPI_COMM_WORLD, \
            FFTW_ESTIMATE); \
 \
    *(FFTW(plan)*)this->r2c_vorticity = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, \
            FFTW_MPI_DEFAULT_BLOCK, \
            FFTW_MPI_DEFAULT_BLOCK, \
            this->rvorticity, this->cvorticity, \
            MPI_COMM_WORLD, \
            FFTW_ESTIMATE); \
 \
    *(FFTW(plan)*)this->c2r_velocity = FFTW(mpi_plan_many_dft_c2r)( \
            3, sizes, 3, \
            FFTW_MPI_DEFAULT_BLOCK, \
            FFTW_MPI_DEFAULT_BLOCK, \
            this->cvelocity, this->rvelocity, \
            MPI_COMM_WORLD, \
            FFTW_ESTIMATE); \
 \
    *(FFTW(plan)*)this->r2c_velocity = FFTW(mpi_plan_many_dft_r2c)( \
            3, sizes, 3, \
            FFTW_MPI_DEFAULT_BLOCK, \
            FFTW_MPI_DEFAULT_BLOCK, \
            this->rvelocity, this->cvelocity, \
            MPI_COMM_WORLD, \
            FFTW_ESTIMATE); \
} \
 \
template<> \
fluid_solver<R>::~fluid_solver() \
{ \
    FFTW(destroy_plan)(*(FFTW(plan)*)this->c2r_vorticity);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->r2c_vorticity);\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->c2r_velocity );\
    FFTW(destroy_plan)(*(FFTW(plan)*)this->r2c_velocity );\
 \
    delete (FFTW(plan)*)this->c2r_vorticity;\
    delete (FFTW(plan)*)this->r2c_vorticity;\
    delete (FFTW(plan)*)this->c2r_velocity ;\
    delete (FFTW(plan)*)this->r2c_velocity ;\
 \
    FFTW(free)(this->cvorticity);\
    FFTW(free)(this->rvorticity);\
    FFTW(free)(this->cvelocity);\
    FFTW(free)(this->rvelocity);\
} \
 \
template<> \
void fluid_solver<R>::step() \
{}
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_DOUBLE, double, fftw_complex)
//FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)
//FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_QUAD, __float128, fftwq_complex)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver<float>;
/*****************************************************************************/

