#include "field_solver.hpp"

#define FLUID_SOLVER_DEFINITIONS(X, R, C) \
 \
X(fluid_solver)::X(fluid_solver)(){} \
 \
X(fluid_solver)::~X(fluid_solver)(){}

FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_DOUBLE, double, fftw_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FLUID_SOLVER_DEFINITIONS(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)

