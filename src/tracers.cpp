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



//#define NDEBUG

#include <cmath>
#include "base.hpp"
#include "fftw_tools.hpp"
#include "tracers.hpp"

template <class rnumber>
void tracers<rnumber>::jump_estimate(double *jump)
{
    DEBUG_MSG("entered jump_estimate\n");
    double *tjump = new double[this->nparticles];
    int *xg = new int[this->array_size];
    double *xx = new double[this->array_size];
    float *vel = this->data + this->buffer_size*this->fs->rd->slice_size;
    double tmp[3];
    /* get grid coordinates */
    this->get_grid_coordinates(this->state, xg, xx);
    DEBUG_MSG("finished get_grid_coordinate\n");

    std::fill_n(tjump, this->nparticles, 0.0);
    /* perform interpolation */
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        DEBUG_MSG("particle %d, about to call linear_interpolation\n", p);
        this->linear_interpolation(vel, xg + p*3, xx + p*3, tmp);
        tjump[p] = fabs(2*this->dt * tmp[2]);
    }
    delete[] xg;
    delete[] xx;
    MPI_Allreduce(
            tjump,
            jump,
            this->nparticles,
            MPI_REAL8,
            MPI_SUM,
            this->fs->rd->comm);
    delete[] tjump;
    DEBUG_MSG("exiting jump_estimate\n");
}

template <class rnumber>
void tracers<rnumber>::get_rhs(double *x, double *y)
{
    std::fill_n(y, this->array_size, 0.0);
    /* get grid coordinates */
    int *xg = new int[this->array_size];
    double *xx = new double[this->array_size];
    float *vel = this->data + this->buffer_size*this->fs->rd->slice_size;
    this->get_grid_coordinates(x, xg, xx);
    /* perform interpolation */
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        this->linear_interpolation(vel, xg + p*3, xx + p*3, y + p*3);
        DEBUG_MSG("particle %d found y %lg %lg %lg\n", p, y[p*3], y[p*3+1], y[p*3+2]);
    }
    delete[] xg;
    delete[] xx;
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

