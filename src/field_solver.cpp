#include "field_solver.hpp"

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

