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

///*****************************************************************************/
///* generic definitions, empty by design                                      */
//
//template <class rnumber>
//fluid_solver<rnumber>::fluid_solver()
//{}
//template <class rnumber>
//fluid_solver<rnumber>::~fluid_solver()
//{}
//template <class rnumber>
//void fluid_solver<rnumber>::step()
//{}
///*****************************************************************************/



/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define FLUID_SOLVER_DEFINITIONS(X, R, C) \
 \
template<> \
fluid_solver<R>::fluid_solver() \
{ \
    this->c2r_vorticity = new X(plan);\
    this->r2c_vorticity = new X(plan);\
    this->c2r_velocity  = new X(plan);\
    this->r2c_velocity  = new X(plan);\
} \
 \
template<> \
fluid_solver<R>::~fluid_solver() \
{ \
    delete (X(plan)*)this->c2r_vorticity;\
    delete (X(plan)*)this->r2c_vorticity;\
    delete (X(plan)*)this->c2r_velocity ;\
    delete (X(plan)*)this->r2c_velocity ;\
} \
 \
template<> \
void fluid_solver<R>::step() \
{}
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_DOUBLE, double, fftw_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_QUAD, __float128, fftwq_complex)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class fluid_solver<float>;
/*****************************************************************************/

