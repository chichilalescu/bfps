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
#include "particles.hpp"
#include "fftw_tools.hpp"


extern int myrank, nprocs;

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
particles<particle_type, rnumber, multistep, interp_neighbours>::particles(
        const char *NAME,
        fluid_solver_base<rnumber> *FSOLVER,
        interpolator<rnumber, interp_neighbours> *FIELD,
        const int NPARTICLES,
        const int TRAJ_SKIP,
        const int INTEGRATION_STEPS)
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->ncomponents = 3;
            break;
        default:
            this->ncomponents = 3;
            break;
    }
    assert((INTEGRATION_STEPS <= 6) &&
           (INTEGRATION_STEPS >= 1));
    strncpy(this->name, NAME, 256);
    this->fs = FSOLVER;
    this->nparticles = NPARTICLES;
    this->vel = FIELD;
    this->integration_steps = INTEGRATION_STEPS;
    this->traj_skip = TRAJ_SKIP;
    // in principle only the buffer width at the top needs the +1,
    // but things are simpler if buffer_width is the same
    this->buffer_width = interp_neighbours+1;
    this->buffer_size = this->buffer_width*this->fs->rd->slice_size;
    this->array_size = this->nparticles * this->ncomponents;
    this->state = fftw_alloc_real(this->array_size);
    std::fill_n(this->state, this->array_size, 0.0);
    for (int i=0; i < this->integration_steps; i++)
    {
        this->rhs[i] = fftw_alloc_real(this->array_size);
        std::fill_n(this->rhs[i], this->array_size, 0.0);
    }
    this->watching = new bool[this->fs->rd->nprocs*nparticles];
    std::fill_n(this->watching, this->fs->rd->nprocs*this->nparticles, false);
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
    //for (int r = 0; r<nprocs; r++)
    //    DEBUG_MSG(
    //            "lbound[%d] = %lg, ubound[%d] = %lg\n",
    //            r, this->lbound[r],
    //            r, this->ubound[r]
    //            );
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
particles<particle_type, rnumber, multistep, interp_neighbours>::~particles()
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

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::get_rhs(double *x, double *y)
{
    std::fill_n(y, this->array_size, 0.0);
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            DEBUG_MSG("aloha from get_rhs\n");
            int deriv[] = {0, 0, 0};
            /* get grid coordinates */
            int *xg = new int[this->array_size];
            double *xx = new double[this->array_size];
            this->get_grid_coordinates(x, xg, xx);
            for (int p=0; p<this->nparticles; p++)
            {
                if (this->watching[this->fs->rd->myrank*this->nparticles+p])
                {
                    int crank = this->get_rank(x[p*3 + 2]);
                    if (this->fs->rd->myrank == crank)
                    {
                        (*this->vel)(xg + p*3, xx + p*3, y + p*3, deriv);
                    }
                    if (crank != this->computing[p])
                    {
                        this->synchronize_single_particle_state(p, y, crank);
                    }
                    //DEBUG_MSG(
                    //        "after synch crank is %d, computing rank is %d, position is %g %g %g, result is %g %g %g\n",
                    //        this->iteration, p,
                    //        crank, this->computing[p],
                    //        x[p*3], x[p*3+1], x[p*3+2],
                    //        y[p*3], y[p*3+1], y[p*3+2]);
                }
            }
            delete[] xg;
            delete[] xx;
            break;
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::jump_estimate(double *jump)
{
    std::fill_n(jump, this->nparticles, 0.0);
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            int deriv[] = {0, 0, 0};
            int *xg = new int[this->array_size];
            double *xx = new double[this->array_size];
            double tmp[3];
            /* get grid coordinates */
            this->get_grid_coordinates(this->state, xg, xx);

            /* perform interpolation */
            for (int p=0; p<this->nparticles; p++) if (this->fs->rd->myrank == this->computing[p])
            {
                (*this->vel)(xg + p*3, xx + p*3, tmp, deriv);
                jump[p] = fabs(3*this->dt * tmp[2]);
                if (jump[p] < this->dz*1.01)
                    jump[p] = this->dz*1.01;
            }
            delete[] xg;
            delete[] xx;
            break;
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
int particles<particle_type, rnumber, multistep, interp_neighbours>::get_rank(double z)
{
    int tmp = this->fs->rd->rank[MOD(int(floor(z/this->dz)), this->fs->rd->sizes[0])];
    assert(tmp >= 0 && tmp < this->fs->rd->nprocs);
    return tmp;
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::synchronize_single_particle_state(int p, double *x, int source)
{
    if (!multistep)
    {
        if (source == -1) source = this->computing[p];
        if (this->watching[this->fs->rd->myrank*this->nparticles+p]) for (int r=0; r<this->fs->rd->nprocs; r++)
            if (r != source &&
                this->watching[r*this->nparticles+p])
            {
                //DEBUG_MSG("synchronizing state %d from %d to %d\n", p, this->computing[p], r);
                if (this->fs->rd->myrank == source)
                    MPI_Send(
                            x+p*this->ncomponents,
                            this->ncomponents,
                            MPI_DOUBLE,
                            r,
                            p+this->computing[p]*this->nparticles,
                            this->fs->rd->comm);
                if (this->fs->rd->myrank == r)
                    MPI_Recv(
                            x+p*this->ncomponents,
                            this->ncomponents,
                            MPI_DOUBLE,
                            source,
                            p+this->computing[p]*this->nparticles,
                            this->fs->rd->comm,
                            MPI_STATUS_IGNORE);
            }
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::synchronize()
{
    double *tstate = fftw_alloc_real(this->array_size);
    // first, synchronize state and jump across CPUs
    std::fill_n(tstate, this->array_size, 0.0);
    for (int p=0; p<this->nparticles; p++)
    {
        //if (this->watching[this->fs->rd->myrank*this->nparticles + p])
        //DEBUG_MSG(
        //        "in synchronize, position for particle %d is %g %g %g\n",
        //        p,
        //        this->state[p*this->ncomponents],
        //        this->state[p*this->ncomponents+1],
        //        this->state[p*this->ncomponents+2]);
        if (this->fs->rd->myrank == this->computing[p])
            std::copy(this->state + p*this->ncomponents,
                      this->state + (p+1)*this->ncomponents,
                      tstate + p*this->ncomponents);
    }
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
    {
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
        for (int r=0; r<this->buffered_field_descriptor->nprocs; r++)
            this->watching[r*this->nparticles+p] = (r == this->computing[p]);
        //DEBUG_MSG("synchronizing particles, particle %d computing is %d\n", p, this->computing[p]);
    }
    if (!multistep)
    {
        double *jump = fftw_alloc_real(this->nparticles);
        this->jump_estimate(jump);
        // now, see who needs to watch
        bool *local_watching = new bool[this->fs->rd->nprocs*this->nparticles];
        std::fill_n(local_watching, this->fs->rd->nprocs*this->nparticles, false);
        for (int p=0; p<this->nparticles; p++)
            if (this->fs->rd->myrank == this->computing[p])
            {
                local_watching[this->get_rank(this->state[this->ncomponents*p+2]        )*this->nparticles+p] = true;
                local_watching[this->get_rank(this->state[this->ncomponents*p+2]-jump[p])*this->nparticles+p] = true;
                local_watching[this->get_rank(this->state[this->ncomponents*p+2]+jump[p])*this->nparticles+p] = true;
            }
        fftw_free(jump);
        MPI_Allreduce(
                local_watching,
                this->watching,
                this->nparticles*this->fs->rd->nprocs,
                MPI_C_BOOL,
                MPI_LOR,
                this->fs->rd->comm);
        delete[] local_watching;
    }
}


template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        std::copy(this->rhs[i],
                  this->rhs[i] + this->array_size,
                  this->rhs[i+1]);
}



template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::AdamsBashforth(int nsteps)
{
    ptrdiff_t ii;
    this->get_rhs(this->state, this->rhs[0]);
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


template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::step()
{
    if (multistep)
        this->AdamsBashforth((this->iteration < this->integration_steps) ?
                                this->iteration+1 :
                                this->integration_steps);
    else
        this->Heun();
    this->iteration++;
    this->synchronize();
}



template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::Heun()
{
    if (!multistep)
    {
        double *y = new double[this->array_size];
        double dtfactor[] = {0.0, this->dt};
        this->get_rhs(this->state, this->rhs[0]);
        for (int p=0; p<this->nparticles; p++)
            this->synchronize_single_particle_state(p, this->rhs[0]);
        for (int kindex = 1; kindex < 2; kindex++)
        {
            for (int p=0; p<this->nparticles; p++)
            {
                if (this->watching[this->fs->rd->myrank*this->nparticles+p])
                    for (int i=0; i<this->ncomponents; i++)
                    {
                        ptrdiff_t tindex = ptrdiff_t(p)*this->ncomponents + i;
                        y[tindex] = this->state[tindex] + dtfactor[kindex]*this->rhs[kindex-1][tindex];
                    }
            }
            for (int p=0; p<this->nparticles; p++)
                this->synchronize_single_particle_state(p, y);
            this->get_rhs(y, this->rhs[kindex]);
            for (int p=0; p<this->nparticles; p++)
                this->synchronize_single_particle_state(p, this->rhs[kindex]);
        }
        for (int p=0; p<this->nparticles; p++)
        {
            if (this->watching[this->fs->rd->myrank*this->nparticles+p])
            {
                for (int i=0; i<this->ncomponents; i++)
                {
                    ptrdiff_t tindex = ptrdiff_t(p)*this->ncomponents + i;
                    this->state[tindex] += this->dt*(this->rhs[0][tindex] + this->rhs[1][tindex])/2;
                }
            }
        }
        delete[] y;
    }
}


template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::cRK4()
{
    if (!multistep)
    {
        double *y = new double[this->array_size];
        double dtfactor[] = {0.0, this->dt/2, this->dt/2, this->dt};
        this->get_rhs(this->state, this->rhs[0]);
        for (int p=0; p<this->nparticles; p++)
            this->synchronize_single_particle_state(p, this->rhs[0]);
        for (int kindex = 1; kindex < 4; kindex++)
        {
            for (int p=0; p<this->nparticles; p++)
            {
                if (this->watching[this->fs->rd->myrank*this->nparticles+p])
                    for (int i=0; i<this->ncomponents; i++)
                    {
                        ptrdiff_t tindex = ptrdiff_t(p)*this->ncomponents + i;
                        y[tindex] = this->state[tindex] + dtfactor[kindex]*this->rhs[kindex-1][tindex];
                    }
            }
            for (int p=0; p<this->nparticles; p++)
                this->synchronize_single_particle_state(p, y);
            this->get_rhs(y, this->rhs[kindex]);
            for (int p=0; p<this->nparticles; p++)
                this->synchronize_single_particle_state(p, this->rhs[kindex]);
        }
        for (int p=0; p<this->nparticles; p++)
        {
            if (this->watching[this->fs->rd->myrank*this->nparticles+p])
                for (int i=0; i<this->ncomponents; i++)
                {
                    ptrdiff_t tindex = ptrdiff_t(p)*this->ncomponents + i;
                    this->state[tindex] += this->dt*(this->rhs[0][tindex] +
                                                  2*(this->rhs[1][tindex] + this->rhs[2][tindex]) +
                                                     this->rhs[3][tindex])/6;
                }
        }
        delete[] y;
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::get_grid_coordinates(double *x, int *xg, double *xx)
{
    static double grid_size[] = {this->dx, this->dy, this->dz};
    double tval;
    std::fill_n(xg, this->nparticles*3, 0);
    std::fill_n(xx, this->nparticles*3, 0.0);
    for (int p=0; p<this->nparticles; p++) if (this->watching[this->fs->rd->myrank*this->nparticles+p])
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
        //DEBUG_MSG(
        //        "particle %d x is %lg %lg %lg xx is %lg %lg %lg xg is %d %d %d\n",
        //        p,
        //         x[p*3],  x[p*3+1],  x[p*3+2],
        //        xx[p*3], xx[p*3+1], xx[p*3+2],
        //        xg[p*3], xg[p*3+1], xg[p*3+2]);
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::read(hid_t data_file_id)
{
    //DEBUG_MSG("aloha\n");
    if (this->fs->rd->myrank == 0)
    {
        std::string temp_string = (std::string("/particles/") +
                                   std::string(this->name) +
                                   std::string("/state"));
        hid_t dset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
        hid_t mspace, rspace;
        hsize_t count[4], offset[4];
        rspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(rspace, count, NULL);
        count[0] = 1;
        offset[0] = this->iteration / this->traj_skip;
        offset[1] = 0;
        offset[2] = 0;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(rspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, rspace, H5P_DEFAULT, this->state);
        H5Sclose(mspace);
        H5Sclose(rspace);
        H5Dclose(dset);
        if (this->iteration > 0)
        {
            temp_string = (std::string("/particles/") +
                           std::string(this->name) +
                           std::string("/rhs"));
            dset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
            rspace = H5Dget_space(dset);
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
                H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, rspace, H5P_DEFAULT, this->rhs[i]);
            }
            H5Sclose(mspace);
            H5Sclose(rspace);
            H5Dclose(dset);
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
    {
        this->computing[p] = this->get_rank(this->state[p*this->ncomponents + 2]);
        //DEBUG_MSG("reading particles, particle %d computing is %d\n", p, this->computing[p]);
    }
    // now actual synchronization
    this->synchronize();
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::write(hid_t data_file_id, bool write_rhs)
{
    if (this->fs->rd->myrank == 0)
    {
        std::string temp_string = (std::string("/particles/") +
                                   std::string(this->name) +
                                   std::string("/state"));
        hid_t dset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
        hid_t mspace, wspace;
        hsize_t count[4], offset[4];
        wspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(wspace, count, NULL);
        count[0] = 1;
        offset[0] = this->iteration / this->traj_skip;
        offset[1] = 0;
        offset[2] = 0;
        mspace = H5Screate_simple(3, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, this->state);
        H5Sclose(mspace);
        H5Sclose(wspace);
        H5Dclose(dset);
        if (write_rhs)
        {
            temp_string = (std::string("/particles/") +
                           std::string(this->name) +
                           std::string("/rhs"));
            dset = H5Dopen(data_file_id, temp_string.c_str(), H5P_DEFAULT);
            wspace = H5Dget_space(dset);
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
                H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, this->rhs[i]);
            }
            H5Sclose(mspace);
            H5Sclose(wspace);
            H5Dclose(dset);
        }
    }
}

template <int particle_type, class rnumber, bool multistep, int interp_neighbours>
void particles<particle_type, rnumber, multistep, interp_neighbours>::sample_vec_field(
        interpolator<rnumber, interp_neighbours> *field,
        double *vec_values)
{
    double *vec_local =  new double[3*this->nparticles];
    std::fill_n(vec_local, 3*this->nparticles, 0.0);
    int deriv[] = {0, 0, 0};
    /* get grid coordinates */
    int *xg = new int[3*this->nparticles];
    double *xx = new double[3*this->nparticles];
    this->get_grid_coordinates(this->state, xg, xx);
    /* perform interpolation */
    for (int p=0; p<this->nparticles; p++)
        if (this->fs->rd->myrank == this->computing[p])
            (*field)(
                    xg + p*3,
                    xx + p*3,
                    vec_local + p*3,
                    deriv);
    MPI_Allreduce(
            vec_local,
            vec_values,
            3*this->nparticles,
            MPI_DOUBLE,
            MPI_SUM,
            this->fs->rd->comm);
    delete[] xg;
    delete[] xx;
    delete[] vec_local;
}


/*****************************************************************************/
template class particles<VELOCITY_TRACER, float, true, 1>;
template class particles<VELOCITY_TRACER, float, true, 2>;
template class particles<VELOCITY_TRACER, float, true, 3>;
template class particles<VELOCITY_TRACER, float, true, 4>;
template class particles<VELOCITY_TRACER, float, true, 5>;
template class particles<VELOCITY_TRACER, float, true, 6>;
template class particles<VELOCITY_TRACER, float, false, 1>;
template class particles<VELOCITY_TRACER, float, false, 2>;
template class particles<VELOCITY_TRACER, float, false, 3>;
template class particles<VELOCITY_TRACER, float, false, 4>;
template class particles<VELOCITY_TRACER, float, false, 5>;
template class particles<VELOCITY_TRACER, float, false, 6>;
template class particles<VELOCITY_TRACER, double, true, 1>;
template class particles<VELOCITY_TRACER, double, true, 2>;
template class particles<VELOCITY_TRACER, double, true, 3>;
template class particles<VELOCITY_TRACER, double, true, 4>;
template class particles<VELOCITY_TRACER, double, true, 5>;
template class particles<VELOCITY_TRACER, double, true, 6>;
template class particles<VELOCITY_TRACER, double, false, 1>;
template class particles<VELOCITY_TRACER, double, false, 2>;
template class particles<VELOCITY_TRACER, double, false, 3>;
template class particles<VELOCITY_TRACER, double, false, 4>;
template class particles<VELOCITY_TRACER, double, false, 5>;
template class particles<VELOCITY_TRACER, double, false, 6>;
/*****************************************************************************/
