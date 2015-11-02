/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/

// code is generally compiled via setuptools, therefore NDEBUG is present
//#ifdef NDEBUG
//#undef NDEBUG
//#endif//NDEBUG


#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>

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
        base_polynomial_values BETA_POLYS,
        const int INTERP_NEIGHBOURS,
        const int TRAJ_SKIP,
        const int INTEGRATION_STEPS)
{
    assert((NCOMPONENTS % 3) == 0);
    assert((INTERP_NEIGHBOURS >= 1) ||
           (INTERP_NEIGHBOURS <= 8));
    assert((INTEGRATION_STEPS <= 6) &&
           (INTEGRATION_STEPS >= 1));
    strncpy(this->name, NAME, 256);
    this->fs = FSOLVER;
    this->nparticles = NPARTICLES;
    this->ncomponents = NCOMPONENTS;
    this->integration_steps = INTEGRATION_STEPS;
    this->interp_neighbours = INTERP_NEIGHBOURS;
    this->traj_skip = TRAJ_SKIP;
    this->compute_beta = BETA_POLYS;
    // in principle only the buffer width at the top needs the +1,
    // but things are simpler if buffer_width is the same
    this->buffer_width = this->interp_neighbours+1;
    this->buffer_size = this->buffer_width*this->fs->rd->slice_size;
    this->array_size = this->nparticles * this->ncomponents;
    this->state = fftw_alloc_real(this->array_size);
    std::fill_n(this->state, this->array_size, 0.0);
    for (int i=0; i < this->integration_steps; i++)
    {
        this->rhs[i] = fftw_alloc_real(this->array_size);
        std::fill_n(this->rhs[i], this->array_size, 0.0);
    }
    this->watching = new bool[nparticles];
    std::fill_n(this->watching, this->nparticles, false);
    this->computing = new int[nparticles];

    int tdims[4];
    tdims[0] = this->buffer_width*2*this->fs->rd->nprocs + this->fs->rd->sizes[0];
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
            MPI_DOUBLE,
            MPI_SUM,
            this->fs->rd->comm);
    std::fill_n(tbound, nprocs, 0.0);
    tbound[this->fs->rd->myrank] = (this->fs->rd->starts[0] + this->fs->rd->subsizes[0])*this->dz;
    MPI_Allreduce(
            tbound,
            this->ubound,
            nprocs,
            MPI_DOUBLE,
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
    for (int i=0; i < this->integration_steps; i++)
    {
        fftw_free(this->rhs[i]);
    }
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
                        MPI_DOUBLE,
                        r,
                        p*this->computing[p],
                        this->fs->rd->comm);
            if (this->fs->rd->myrank == r)
                MPI_Recv(
                        this->state + p*this->ncomponents,
                        this->ncomponents,
                        MPI_DOUBLE,
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
    // first, synchronize state and jump across CPUs
    std::fill_n(tstate, this->array_size, 0.0);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
            std::copy(this->state + p*this->ncomponents,
                      this->state + (p+1)*this->ncomponents,
                      tstate + p*this->ncomponents);
    std::fill_n(this->state, this->array_size, 0.0);
    MPI_Allreduce(
            tstate,
            this->state,
            this->array_size,
            MPI_DOUBLE,
            MPI_SUM,
            this->fs->rd->comm);
    if (this->integration_steps >= 1)
    {
        for (int i=0; i<this->integration_steps; i++)
        {
            std::fill_n(tstate, this->array_size, 0.0);
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                std::copy(this->rhs[i] + p*this->ncomponents,
                          this->rhs[i] + (p+1)*this->ncomponents,
                          tstate + p*this->ncomponents);
            std::fill_n(this->rhs[i], this->array_size, 0.0);
            MPI_Allreduce(
                    tstate,
                    this->rhs[i],
                    this->array_size,
                    MPI_DOUBLE,
                    MPI_SUM,
                    this->fs->rd->comm);
        }
    }
    fftw_free(tstate);
    // assignment of particles
    for (int p=0; p<this->nparticles; p++)
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
    double *jump = fftw_alloc_real(this->nparticles);
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
void slab_field_particles<rnumber>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        std::copy(this->rhs[i],
                  this->rhs[i] + this->array_size,
                  this->rhs[i+1]);
}



template <class rnumber>
void slab_field_particles<rnumber>::AdamsBashforth(int nsteps)
{
    ptrdiff_t ii;
    this->get_rhs(this->state, this->rhs[0]);
    //if (myrank == 0)
    //{
    //    DEBUG_MSG(
    //            "in AdamsBashforth for particles %s, integration_steps = %d, nsteps = %d, iteration = %d\n",
    //            this->name,
    //            this->integration_steps,
    //            nsteps,
    //            this->iteration);
    //    std::stringstream tstring;
    //    for (int p=0; p<this->nparticles; p++)
    //        tstring << " " << this->computing[p];
    //    DEBUG_MSG("%s\n", tstring.str().c_str());
    //    for (int i=0; i<this->integration_steps; i++)
    //    {
    //        std::stringstream tstring;
    //        for (int p=0; p<this->nparticles; p++)
    //            tstring << " " << this->rhs[i][p*3];
    //        DEBUG_MSG("%s\n", tstring.str().c_str());
    //    }
    //}
    switch(nsteps)
    {
        case 1:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*this->rhs[0][ii];
                }
            break;
        case 2:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(3*this->rhs[0][ii]
                                               -   this->rhs[1][ii])/2;
                }
            break;
        case 3:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(23*this->rhs[0][ii]
                                               - 16*this->rhs[1][ii]
                                               +  5*this->rhs[2][ii])/12;
                }
            break;
        case 4:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(55*this->rhs[0][ii]
                                               - 59*this->rhs[1][ii]
                                               + 37*this->rhs[2][ii]
                                               -  9*this->rhs[3][ii])/24;
                }
            break;
        case 5:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(1901*this->rhs[0][ii]
                                               - 2774*this->rhs[1][ii]
                                               + 2616*this->rhs[2][ii]
                                               - 1274*this->rhs[3][ii]
                                               +  251*this->rhs[4][ii])/720;
                }
            break;
        case 6:
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(4277*this->rhs[0][ii]
                                               - 7923*this->rhs[1][ii]
                                               + 9982*this->rhs[2][ii]
                                               - 7298*this->rhs[3][ii]
                                               + 2877*this->rhs[4][ii]
                                               -  475*this->rhs[5][ii])/1440;
                }
            break;
    }
    this->roll_rhs();
}


template <class rnumber>
void slab_field_particles<rnumber>::step()
{
    this->AdamsBashforth((this->iteration < this->integration_steps) ? this->iteration+1 : this->integration_steps);
    //this->cRK4();
    this->iteration++;
    this->synchronize();
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
        //DEBUG_MSG(
        //        "particle %d state is %lg %lg %lg\n",
        //        p, this->state[p*this->ncomponents], this->state[p*this->ncomponents+1], this->state[p*this->ncomponents+2]);
    }
    fftw_free(y);
}


template <class rnumber>
void slab_field_particles<rnumber>::cRK4()
{
    double *k1 = new double[this->array_size];
    double *k2 = new double[this->array_size];
    double *k3 = new double[this->array_size];
    double *k4 = new double[this->array_size];
    double *y = new double[this->array_size];
    this->get_rhs(this->state, k1);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int i=0; i<this->ncomponents; i++)
            y[p*this->ncomponents+i] = this->state[p*this->ncomponents+i] + this->dt*k1[p*this->ncomponents+i]/2;
    }
    this->get_rhs(y, k2);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int i=0; i<this->ncomponents; i++)
            y[p*this->ncomponents+i] = this->state[p*this->ncomponents+i] + this->dt*k2[p*this->ncomponents+i]/2;
    }
    this->get_rhs(y, k3);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int i=0; i<this->ncomponents; i++)
            y[p*this->ncomponents+i] = this->state[p*this->ncomponents+i] + this->dt*k3[p*this->ncomponents+i];
    }
    this->get_rhs(y, k4);
    for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
    {
        for (int i=0; i<this->ncomponents; i++)
            this->state[p*this->ncomponents+i] += this->dt*(k1[p*this->ncomponents+i] +
                                                            k4[p*this->ncomponents+i] +
                                                            2*(k2[p*this->ncomponents+i] +
                                                               k3[p*this->ncomponents+i]))/6;
    }
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;
    delete[] y;
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
void slab_field_particles<rnumber>::spline_formula(rnumber *field, int *xg, double *xx, double *dest, int *deriv)
{
    double bx[this->interp_neighbours*2+2], by[this->interp_neighbours*2+2], bz[this->interp_neighbours*2+2];
    //DEBUG_MSG("entered spline_formula\n");
    this->compute_beta(deriv[0], xx[0], bx);
    this->compute_beta(deriv[1], xx[1], by);
    this->compute_beta(deriv[2], xx[2], bz);
    //DEBUG_MSG("computed beta polynomials\n");
    std::fill_n(dest, 3, 0);
    for (int iz = -this->interp_neighbours; iz <= this->interp_neighbours+1; iz++)
    for (int iy = -this->interp_neighbours; iy <= this->interp_neighbours+1; iy++)
    for (int ix = -this->interp_neighbours; ix <= this->interp_neighbours+1; ix++)
        for (int c=0; c<3; c++)
        {
            //DEBUG_MSG(
            //        "%d %d %d %d %d %d %d %ld %ld\n",
            //        xg[2], xg[1], xg[0], iz, iy, ix, c,
            //        ((ptrdiff_t(xg[2]+iz) *this->fs->rd->subsizes[1] +
            //          ptrdiff_t(xg[1]+iy))*this->fs->rd->subsizes[2] +
            //          ptrdiff_t(xg[0]+ix))*3+c,
            //        this->buffered_field_descriptor->local_size
            //        );
            dest[c] += field[((ptrdiff_t(    xg[2]+iz                         ) *this->fs->rd->subsizes[1] +
                               ptrdiff_t(MOD(xg[1]+iy, this->fs->rd->sizes[1])))*this->fs->rd->subsizes[2] +
                               ptrdiff_t(MOD(xg[0]+ix, this->fs->rd->sizes[2])))*3+c]*(bz[iz+this->interp_neighbours]*
                                                                                       by[iy+this->interp_neighbours]*
                                                                                       bx[ix+this->interp_neighbours]);
        }
    //DEBUG_MSG("finishing spline_formula\n");
}

template <class rnumber>
void slab_field_particles<rnumber>::spline_n1_formula(rnumber *field, int *xg, double *xx, double *dest, int *deriv)
{
    double bx[4], by[4], bz[4];
    this->compute_beta(deriv[0], xx[0], bx);
    this->compute_beta(deriv[1], xx[1], by);
    this->compute_beta(deriv[2], xx[2], bz);
    std::fill_n(dest, 3, 0);
    for (int iz = -1; iz <= 2; iz++)
    for (int iy = -1; iy <= 2; iy++)
    for (int ix = -1; ix <= 2; ix++)
        for (int c=0; c<3; c++)
            dest[c] += field[((ptrdiff_t(    xg[2]+iz                         ) *this->fs->rd->subsizes[1] +
                               ptrdiff_t(MOD(xg[1]+iy, this->fs->rd->sizes[1])))*this->fs->rd->subsizes[2] +
                               ptrdiff_t(MOD(xg[0]+ix, this->fs->rd->sizes[2])))*3+c]*bz[iz+1]*by[iy+1]*bx[ix+1];
}

template <class rnumber>
void slab_field_particles<rnumber>::spline_n2_formula(rnumber *field, int *xg, double *xx, double *dest, int *deriv)
{
    double bx[6], by[6], bz[6];
    this->compute_beta(deriv[0], xx[0], bx);
    this->compute_beta(deriv[1], xx[1], by);
    this->compute_beta(deriv[2], xx[2], bz);
    std::fill_n(dest, 3, 0);
    for (int iz = -2; iz <= 3; iz++)
    for (int iy = -2; iy <= 3; iy++)
    for (int ix = -2; ix <= 3; ix++)
        for (int c=0; c<3; c++)
            dest[c] += field[((ptrdiff_t(xg[2]+iz) *this->fs->rd->subsizes[1] +
                               ptrdiff_t(xg[1]+iy))*this->fs->rd->subsizes[2] +
                               ptrdiff_t(xg[0]+ix))*3+c]*bz[iz+2]*by[iy+2]*bx[ix+2];
}

template <class rnumber>
void slab_field_particles<rnumber>::spline_n3_formula(rnumber *field, int *xg, double *xx, double *dest, int *deriv)
{
    double bx[8], by[8], bz[8];
    this->compute_beta(deriv[0], xx[0], bx);
    this->compute_beta(deriv[1], xx[1], by);
    this->compute_beta(deriv[2], xx[2], bz);
    std::fill_n(dest, 3, 0);
    for (int iz = -3; iz <= 4; iz++)
    for (int iy = -3; iy <= 4; iy++)
    for (int ix = -3; ix <= 4; ix++)
        for (int c=0; c<3; c++)
            dest[c] += field[((ptrdiff_t(xg[2]+iz) *this->fs->rd->subsizes[1] +
                               ptrdiff_t(xg[1]+iy))*this->fs->rd->subsizes[2] +
                               ptrdiff_t(xg[0]+ix))*3+c]*bz[iz+3]*by[iy+3]*bx[ix+3];
}

template <class rnumber>
void slab_field_particles<rnumber>::linear_interpolation(rnumber *field, int *xg, double *xx, double *dest, int *deriv)
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
void slab_field_particles<rnumber>::read(hid_t data_file_id)
{
    DEBUG_MSG("aloha\n");
    if (this->fs->rd->myrank == 0)
    {
        std::string temp_string = (std::string("/particles/") +
                                   std::string(this->name) +
                                   std::string("/state"));
        hid_t Cdset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
        hid_t mspace, rspace;
        hsize_t count[4], offset[4];
        rspace = H5Dget_space(Cdset);
        H5Sget_simple_extent_dims(rspace, count, NULL);
        count[0] = 1;
        offset[0] = this->iteration / this->traj_skip;
        offset[1] = 0;
        offset[2] = 0;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(rspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dread(Cdset, H5T_NATIVE_DOUBLE, mspace, rspace, H5P_DEFAULT, this->state);
        H5Sclose(mspace);
        H5Sclose(rspace);
        H5Dclose(Cdset);
        if (this->iteration > 0)
        {
            temp_string = (std::string("/particles/") +
                           std::string(this->name) +
                           std::string("/rhs"));
            Cdset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
            rspace = H5Dget_space(Cdset);
            H5Sget_simple_extent_dims(rspace, count, NULL);
            //reading from last available position
            offset[0] = count[0] - 1;
            offset[3] = 0;
            count[0] = 1;
            count[1] = 1;
            mspace = H5Screate_simple(4, count, NULL);
            for (int i=0; i<this->integration_steps; i++)
            {
                offset[1] = i;
                H5Sselect_hyperslab(rspace, H5S_SELECT_SET, offset, NULL, count, NULL);
                H5Dread(Cdset, H5T_NATIVE_DOUBLE, mspace, rspace, H5P_DEFAULT, this->rhs[i]);
            }
            H5Sclose(mspace);
            H5Sclose(rspace);
            H5Dclose(Cdset);
        }
    }
    MPI_Bcast(
            this->state,
            this->array_size,
            MPI_DOUBLE,
            0,
            this->fs->rd->comm);
    for (int i = 0; i<this->integration_steps; i++)
    {
        MPI_Bcast(
                this->rhs[i],
                this->array_size,
                MPI_DOUBLE,
                0,
                this->fs->rd->comm);
    }
    // initial assignment of particles
    for (int p=0; p<this->nparticles; p++)
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
    // now actual synchronization
    this->synchronize();
}

template <class rnumber>
void slab_field_particles<rnumber>::write(hid_t data_file_id, bool write_rhs)
{
    if (this->fs->rd->myrank == 0)
    {
        std::string temp_string = (std::string("/particles/") +
                                   std::string(this->name) +
                                   std::string("/state"));
        hid_t Cdset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
        hid_t mspace, wspace;
        hsize_t count[4], offset[4];
        wspace = H5Dget_space(Cdset);
        H5Sget_simple_extent_dims(wspace, count, NULL);
        count[0] = 1;
        offset[0] = this->iteration / this->traj_skip;
        offset[1] = 0;
        offset[2] = 0;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, this->state);
        H5Sclose(mspace);
        H5Sclose(wspace);
        H5Dclose(Cdset);
        if (write_rhs)
        {
            temp_string = (std::string("/particles/") +
                           std::string(this->name) +
                           std::string("/rhs"));
            Cdset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
            wspace = H5Dget_space(Cdset);
            H5Sget_simple_extent_dims(wspace, count, NULL);
            //writing to last available position
            offset[0] = count[0] - 1;
            count[0] = 1;
            count[1] = 1;
            offset[3] = 0;
            mspace = H5Screate_simple(4, count, NULL);
            for (int i=0; i<this->integration_steps; i++)
            {
                offset[1] = i;
                H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
                H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, this->rhs[i]);
            }
            H5Sclose(mspace);
            H5Sclose(wspace);
            H5Dclose(Cdset);
        }
    }
}



/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */
#define SLAB_FIELD_PARTICLES_DEFINITIONS(FFTW, R, MPI_RNUM) \
 \
template <> \
void slab_field_particles<R>::rFFTW_to_buffered(R *src, R *dst) \
{ \
    /* do big copy of middle stuff */ \
    std::copy(src, \
              src + this->fs->rd->local_size, \
              dst + this->buffer_size); \
    int rsrc; \
    /* get upper slices */ \
    for (int rdst = 0; rdst < this->fs->rd->nprocs; rdst++) \
    { \
        rsrc = this->fs->rd->rank[(this->fs->rd->all_start0[rdst] + \
                                   this->fs->rd->all_size0[rdst]) % \
                                   this->fs->rd->sizes[0]]; \
        if (this->fs->rd->myrank == rsrc) \
            MPI_Send( \
                    (void*)(src), \
                    this->buffer_size, \
                    MPI_RNUM, \
                    rdst, \
                    2*(rsrc*this->fs->rd->nprocs + rdst), \
                    this->fs->rd->comm); \
        if (this->fs->rd->myrank == rdst) \
            MPI_Recv( \
                    (void*)(dst + this->buffer_size + this->fs->rd->local_size), \
                    this->buffer_size, \
                    MPI_RNUM, \
                    rsrc, \
                    2*(rsrc*this->fs->rd->nprocs + rdst), \
                    this->fs->rd->comm, \
                    MPI_STATUS_IGNORE); \
    } \
    /* get lower slices */ \
    for (int rdst = 0; rdst < this->fs->rd->nprocs; rdst++) \
    { \
        rsrc = this->fs->rd->rank[MOD(this->fs->rd->all_start0[rdst] - 1, \
                                      this->fs->rd->sizes[0])]; \
        if (this->fs->rd->myrank == rsrc) \
            MPI_Send( \
                    (void*)(src + this->fs->rd->local_size - this->buffer_size), \
                    this->buffer_size, \
                    MPI_RNUM, \
                    rdst, \
                    2*(rsrc*this->fs->rd->nprocs + rdst)+1, \
                    this->fs->rd->comm); \
        if (this->fs->rd->myrank == rdst) \
            MPI_Recv( \
                    (void*)(dst), \
                    this->buffer_size, \
                    MPI_RNUM, \
                    rsrc, \
                    2*(rsrc*this->fs->rd->nprocs + rdst)+1, \
                    this->fs->rd->comm, \
                    MPI_STATUS_IGNORE); \
    } \
} \
/*****************************************************************************/



/*****************************************************************************/
/* now actually use the macro defined above                                  */
SLAB_FIELD_PARTICLES_DEFINITIONS(
        FFTW_MANGLE_FLOAT,
        float,
        MPI_FLOAT)
SLAB_FIELD_PARTICLES_DEFINITIONS(
        FFTW_MANGLE_DOUBLE,
        double,
        MPI_DOUBLE)
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code for single precision                    */
template class slab_field_particles<float>;
template class slab_field_particles<double>;
/*****************************************************************************/
