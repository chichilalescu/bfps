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



#include "fftw_tools.hpp"
#include "tracers.hpp"

template <class rnumber>
void tracers<rnumber>::jump_estimate(double *jump)
{
    std::fill_n(jump, this->nparticles, 0.0);
}

template <class rnumber>
void tracers<rnumber>::get_rhs(double *x, double *y)
{
    /* get grid coordinates */
    /* perform interpolation */
}

template<class rnumber>
void tracers<rnumber>::update_field(bool clip_on)
{
    if (clip_on)
        clip_zero_padding(this->fs->rd, this->source_data, 3);
    this->rFFTW_to_buffered(this->source_data, this->data);
}

/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */

#define TRACERS_DEFINITIONS(FFTW, R, C, MPI_RNUM, MPI_CNUM) \
 \
template <> \
tracers<R>::tracers( \
                const char *NAME, \
                fluid_solver_base<R> *FSOLVER, \
                const int NPARTICLES, \
                const int BUFFERSIZE, \
                R *SOURCE_DATA) : slab_field_particles<R>( \
                    NAME, \
                    FSOLVER, \
                    NPARTICLES, \
                    3, \
                    BUFFERSIZE) \
{ \
    this->source_data = SOURCE_DATA; \
    this->data = FFTW(alloc_real)(this->buffered_local_size()); \
} \
 \
template<> \
tracers<R>::~tracers() \
{ \
    FFTW(free)(this->data); \
}
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
TRACERS_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        fftwf_complex,
        MPI_REAL4,
        MPI_COMPLEX8)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class tracers<float>;
/*****************************************************************************/

