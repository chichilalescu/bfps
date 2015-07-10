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



#define NDEBUG

#include <cmath>
#include <cassert>
#include <cstring>
#include "base.hpp"
#include "slab_field_particles.hpp"
#include "fftw_tools.hpp"

extern int myrank, nprocs;

template <class rnumber>
slab_field_particles<rnumber>::slab_field_particles(
        const char *NAME,
        fluid_solver_base<rnumber> *FSOLVER,
        const int NPARTICLES,
        const int NCOMPONENTS,
        const int BUFFERSIZE)
{
    assert((NCOMPONENTS % 3) == 0);
    strncpy(this->name, NAME, 256);
    this->fs = FSOLVER;
    this->nparticles = NPARTICLES;
    this->ncomponents = NCOMPONENTS;
    this->buffer_size = BUFFERSIZE;
    this->array_size = this->nparticles * this->ncomponents;
    this->state = fftw_alloc_real(this->array_size);
    std::fill_n(this->state, this->array_size, 0.0);
    this->watching = new bool[nparticles];
    std::fill_n(this->watching, this->nparticles, false);
    this->computing = new int[nparticles];

    int tdims[4];
    tdims[0] = this->buffer_size*2*this->fs->rd->nprocs + this->fs->rd->sizes[0];
    tdims[1] = this->fs->rd->sizes[1];
    tdims[2] = this->fs->rd->sizes[2];
    tdims[3] = this->fs->rd->sizes[3];
    this->buffered_field_descriptor = new field_descriptor<rnumber>(
            4, tdims,
            this->fs->rd->mpi_dtype,
            this->fs->rd->comm);

    // compute dx, dy, dz;
    this->dx = 4*acos(0) / (this->fs->dkx*this->fs->rd->sizes[2]);
    this->dy = 4*acos(0) / (this->fs->dky*this->fs->rd->sizes[1]);
    this->dz = 4*acos(0) / (this->fs->dkz*this->fs->rd->sizes[0]);

    // compute lower and upper bounds
    this->lbound = new double[nprocs];
    this->ubound = new double[nprocs];
    double *tbound = new double[nprocs];
    std::fill_n(tbound, nprocs, 0.0);
    tbound[this->fs->rd->myrank] = this->fs->rd->starts[0]*this->dz;
    MPI_Allreduce(
            tbound,
            this->lbound,
            nprocs,
            MPI_REAL8,
            MPI_SUM,
            this->fs->rd->comm);
    std::fill_n(tbound, nprocs, 0.0);
    tbound[this->fs->rd->myrank] = (this->fs->rd->starts[0] + this->fs->rd->subsizes[0])*this->dz;
    MPI_Allreduce(
            tbound,
            this->ubound,
            nprocs,
            MPI_REAL8,
            MPI_SUM,
            this->fs->rd->comm);
    delete[] tbound;
    for (int r = 0; r<nprocs; r++)
        DEBUG_MSG(
                "lbound[%d] = %lg, ubound[%d] = %lg\n",
                r, this->lbound[r],
                r, this->ubound[r]
                );
}

template <class rnumber>
slab_field_particles<rnumber>::~slab_field_particles()
{
    delete[] this->computing;
    delete[] this->watching;
    fftw_free(this->state);
    delete[] this->lbound;
    delete[] this->ubound;
    delete this->buffered_field_descriptor;
}

template <class rnumber>
void slab_field_particles<rnumber>::get_rhs(double *x, double *y)
{
    std::fill_n(y, this->array_size, 0.0);
}

template <class rnumber>
void slab_field_particles<rnumber>::jump_estimate(double *dest)
{
    std::fill_n(dest, this->nparticles, 0.0);
}

template <class rnumber>
int slab_field_particles<rnumber>::get_rank(double z)
{
    int tmp = this->fs->rd->rank[MOD(floor(z/this->dz), this->fs->rd->sizes[0])];
    assert(tmp >= 0 && tmp < this->fs->rd->sizes[0]);
    return tmp;
}

template <class rnumber>
void slab_field_particles<rnumber>::synchronize_single_particle(int p)
{
    MPI_Status *s = new MPI_Status;
    if (this->watching[p]) for (int r=0; r<this->fs->rd->nprocs; r++)
        if (r != this->computing[p])
        {
            if (this->fs->rd->myrank == this->computing[p])
                MPI_Send(
                        this->state + p*this->ncomponents,
                        this->ncomponents,
                        MPI_REAL8,
                        r,
                        p*this->computing[p],
                        this->fs->rd->comm);
            if (this->fs->rd->myrank == r)
                MPI_Recv(
                        this->state + p*this->ncomponents,
                        this->ncomponents,
                        MPI_REAL8,
                        this->computing[p],
                        p*this->computing[p],
                        this->fs->rd->comm,
                        s);
        }
    delete s;
}

template <class rnumber>
void slab_field_particles<rnumber>::synchronize()
{
    double *tstate = fftw_alloc_real(this->array_size);
    double *jump = fftw_alloc_real(this->nparticles);
    // first, synchronize state and jump across CPUs
    std::fill_n(tstate, this->array_size, 0.0);
    for (int p=0; p<this->nparticles; p++)
    {
        if (this->fs->rd->myrank == this->computing[p])
            std::copy(this->state + p*this->ncomponents,
                      this->state + (p+1)*this->ncomponents,
                      tstate + p*this->ncomponents);
    }
    std::fill_n(this->state, this->array_size, 0.0);
    MPI_Allreduce(
            tstate,
            this->state,
            this->array_size,
            MPI_REAL8,
            MPI_SUM,
            this->fs->rd->comm);
    fftw_free(tstate);
    // assignment of particles
    for (int p=0; p<this->nparticles; p++)
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
    this->jump_estimate(jump);
    // now, see who needs to watch
    bool *local_watching = new bool[this->nparticles];
    std::fill_n(local_watching, this->nparticles, false);
    for (int p=0; p<this->nparticles; p++)
        if (this->fs->rd->myrank == this->computing[p])
        {
            local_watching[this->get_rank(this->state[this->ncomponents*p+2])] = true;
            local_watching[this->get_rank(this->state[this->ncomponents*p+2]-jump[p])] = true;
            local_watching[this->get_rank(this->state[this->ncomponents*p+2]+jump[p])] = true;
        }
    fftw_free(jump);
    MPI_Allreduce(
            local_watching,
            this->watching,
            this->nparticles,
            MPI_C_BOOL,
            MPI_LOR,
            this->fs->rd->comm);
    delete[] local_watching;
}



template <class rnumber>
void slab_field_particles<rnumber>::Euler()
{
    double *y = fftw_alloc_real(this->array_size);
    this->get_rhs(this->state, y);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int i=0; i<this->ncomponents; i++)
            this->state[p*this->ncomponents+i] += this->dt*y[p*this->ncomponents+i];
        DEBUG_MSG(
                "particle %d state is %lg %lg %lg\n",
                p, this->state[p*this->ncomponents], this->state[p*this->ncomponents+1], this->state[p*this->ncomponents+2]);
    }
    fftw_free(y);
}

template <class rnumber>
void slab_field_particles<rnumber>::get_grid_coordinates(double *x, int *xg, double *xx)
{
    static double grid_size[] = {this->dx, this->dy, this->dz};
    double tval;
    std::fill_n(xg, this->nparticles*3, 0);
    std::fill_n(xx, this->nparticles*3, 0.0);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int c=0; c<3; c++)
        {
            tval = floor(x[p*this->ncomponents+c]/grid_size[c]);
            xg[p*3+c] = MOD(int(tval), this->fs->rd->sizes[2-c]);
            xx[p*3+c] = (x[p*this->ncomponents+c] - tval*grid_size[c]) / grid_size[c];
        }
        xg[p*3+2] -= this->fs->rd->starts[0];
        if (this->fs->rd->myrank == this->fs->rd->rank[0] &&
            xg[p*3+2] > this->fs->rd->subsizes[0])
            xg[p*3+2] -= this->fs->rd->sizes[0];
        DEBUG_MSG(
                "particle %d x is %lg %lg %lg xx is %lg %lg %lg xg is %d %d %d\n",
                p,
                 x[p*3],  x[p*3+1],  x[p*3+2],
                xx[p*3], xx[p*3+1], xx[p*3+2],
                xg[p*3], xg[p*3+1], xg[p*3+2]);
    }
}

template <class rnumber>
void slab_field_particles<rnumber>::linear_interpolation(float *field, int *xg, double *xx, double *dest)
{
    //ptrdiff_t tindex, tmp;
    //tindex = ((ptrdiff_t(xg[2]  )*this->fs->rd->subsizes[1]+xg[1]  )*this->fs->rd->subsizes[2]+xg[0]  )*3;
    //tmp = ptrdiff_t(xg[2]);
    //DEBUG_MSG(
    //        "linear interpolation xx is %lg %lg %lg xg is %d %d %d,"
    //        " corner index is ((%ld*%d+%d)*%d+%d)*3 = %ld\n",
    //        xx[0], xx[1], xx[2],
    //        xg[0], xg[1], xg[2],
    //        tmp, this->fs->rd->subsizes[1], xg[1], this->fs->rd->subsizes[2], xg[0],
    //        tindex);
    for (int c=0; c<3; c++)
        dest[c] = (field[((ptrdiff_t(xg[2]  )*this->fs->rd->subsizes[1]+xg[1]  )*this->fs->rd->subsizes[2]+xg[0]  )*3+c]*((1-xx[0])*(1-xx[1])*(1-xx[2])) +
                   field[((ptrdiff_t(xg[2]  )*this->fs->rd->subsizes[1]+xg[1]  )*this->fs->rd->subsizes[2]+xg[0]+1)*3+c]*((  xx[0])*(1-xx[1])*(1-xx[2])) +
                   field[((ptrdiff_t(xg[2]  )*this->fs->rd->subsizes[1]+xg[1]+1)*this->fs->rd->subsizes[2]+xg[0]  )*3+c]*((1-xx[0])*(  xx[1])*(1-xx[2])) +
                   field[((ptrdiff_t(xg[2]  )*this->fs->rd->subsizes[1]+xg[1]+1)*this->fs->rd->subsizes[2]+xg[0]+1)*3+c]*((  xx[0])*(  xx[1])*(1-xx[2])) +
                   field[((ptrdiff_t(xg[2]+1)*this->fs->rd->subsizes[1]+xg[1]  )*this->fs->rd->subsizes[2]+xg[0]  )*3+c]*((1-xx[0])*(1-xx[1])*(  xx[2])) +
                   field[((ptrdiff_t(xg[2]+1)*this->fs->rd->subsizes[1]+xg[1]  )*this->fs->rd->subsizes[2]+xg[0]+1)*3+c]*((  xx[0])*(1-xx[1])*(  xx[2])) +
                   field[((ptrdiff_t(xg[2]+1)*this->fs->rd->subsizes[1]+xg[1]+1)*this->fs->rd->subsizes[2]+xg[0]  )*3+c]*((1-xx[0])*(  xx[1])*(  xx[2])) +
                   field[((ptrdiff_t(xg[2]+1)*this->fs->rd->subsizes[1]+xg[1]+1)*this->fs->rd->subsizes[2]+xg[0]+1)*3+c]*((  xx[0])*(  xx[1])*(  xx[2])));
}

template <class rnumber>
void slab_field_particles<rnumber>::read()
{
    std::fill_n(this->state, this->array_size, 0.0);
    if (this->fs->rd->myrank == 0)
    {
        char full_name[512];
        sprintf(full_name, "%s_state_i%.5x", this->name, this->iteration);
        FILE *ifile;
        ifile = fopen(full_name, "rb");
        fread((void*)this->state, sizeof(double), this->array_size, ifile);
        fclose(ifile);
    }
    MPI_Bcast(
            this->state,
            this->array_size,
            MPI_REAL8,
            0,
            this->fs->rd->comm);
    // initial assignment of particles
    for (int p=0; p<this->nparticles; p++)
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
    // now actual synchronization
    this->synchronize();
}

template <class rnumber>
void slab_field_particles<rnumber>::write()
{
    this->synchronize();
    if (this->fs->rd->myrank == 0)
    {
        char full_name[512];
        sprintf(full_name, "%s_state_i%.5x", this->name, this->iteration);
        FILE *ofile;
        ofile = fopen(full_name, "wb");
        fwrite((void*)this->state, sizeof(double), this->array_size, ofile);
        fclose(ofile);
    }
}

template <class rnumber>
ptrdiff_t slab_field_particles<rnumber>::buffered_local_size()
{
    return this->fs->rd->local_size + this->buffer_size*2*this->fs->rd->slice_size;
}

template <class rnumber>
void slab_field_particles<rnumber>::rFFTW_to_buffered(rnumber *src, rnumber *dst)
{
    const MPI_Datatype MPI_RNUM = (sizeof(rnumber) == 4) ? MPI_REAL4 : MPI_REAL8;
    const ptrdiff_t bsize = this->buffer_size*this->fs->rd->slice_size;
    MPI_Request *mpirequest = new MPI_Request;
    /* do big copy of middle stuff */
    std::copy(src,
              src + this->fs->rd->local_size,
              dst + bsize);
    /* take care of buffer regions.
     * I could make the code use blocking sends and receives, but it seems cleaner this way.
     * (alternative is to have a couple of loops).
     * */
    // 1. send lower slices
    //DEBUG_MSG(
    //        "destination rank is %d, message is %d\n",
    //        this->fs->rd->rank[MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0])],
    //        MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0]));
    MPI_Isend(
            (void*)(src),
            bsize,
            MPI_RNUM,
            this->fs->rd->rank[MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0])],
            MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0]),
            this->fs->rd->comm,
            mpirequest);
    // 2. receive higher slices
    //DEBUG_MSG(
    //        "source rank is %d, message is %d\n",
    //        this->fs->rd->rank[MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0])],
    //        MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0]-1, this->fs->rd->sizes[0]));
    MPI_Irecv(
            (void*)(dst + bsize + this->fs->rd->local_size),
            bsize,
            MPI_RNUM,
            this->fs->rd->rank[MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0])],
            MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0]-1, this->fs->rd->sizes[0]),
            this->fs->rd->comm,
            mpirequest);
    //DEBUG_MSG("successful transfer\n");
    // 3. send higher slices
    //DEBUG_MSG(
    //        "destination rank is %d, message is %d\n",
    //        this->fs->rd->rank[MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0])],
    //        MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0]));
    MPI_Isend(
            (void*)(src + this->fs->rd->local_size - bsize),
            bsize,
            MPI_RNUM,
            this->fs->rd->rank[MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0])],
            MOD(this->fs->rd->starts[0]+this->fs->rd->subsizes[0], this->fs->rd->sizes[0]),
            this->fs->rd->comm,
            mpirequest);
    // 4. receive lower slices
    //DEBUG_MSG(
    //        "source rank is %d, message is %d\n",
    //        this->fs->rd->rank[MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0])],
    //        this->fs->rd->starts[0]);
    MPI_Irecv(
            (void*)(dst),
            bsize,
            MPI_RNUM,
            this->fs->rd->rank[MOD(this->fs->rd->starts[0]-1, this->fs->rd->sizes[0])],
            this->fs->rd->starts[0],
            this->fs->rd->comm,
            mpirequest);
    delete mpirequest;
}

/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class slab_field_particles<float>;
/*****************************************************************************/
