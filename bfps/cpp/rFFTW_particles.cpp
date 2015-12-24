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



#define NDEBUG

#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>

#include "base.hpp"
#include "rFFTW_particles.hpp"
#include "fftw_tools.hpp"


extern int myrank, nprocs;

template <int particle_type, class rnumber, int interp_neighbours>
rFFTW_particles<particle_type, rnumber, interp_neighbours>::rFFTW_particles(
        const char *NAME,
        fluid_solver_base<rnumber> *FSOLVER,
        rFFTW_interpolator<rnumber, interp_neighbours> *FIELD,
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
    this->myrank = this->vel->descriptor->myrank;
    this->nprocs = this->vel->descriptor->nprocs;
    this->comm = this->vel->descriptor->comm;
    this->array_size = this->nparticles * this->ncomponents;
    this->state = new double[this->array_size];
    std::fill_n(this->state, this->array_size, 0.0);
    for (int i=0; i < this->integration_steps; i++)
    {
        this->rhs[i] = new double[this->array_size];
        std::fill_n(this->rhs[i], this->array_size, 0.0);
    }

    // compute dx, dy, dz;
    this->dx = 4*acos(0) / (this->fs->dkx*this->fs->rd->sizes[2]);
    this->dy = 4*acos(0) / (this->fs->dky*this->fs->rd->sizes[1]);
    this->dz = 4*acos(0) / (this->fs->dkz*this->fs->rd->sizes[0]);

    // compute lower and upper bounds
    this->lbound = new double[nprocs];
    this->ubound = new double[nprocs];
    double *tbound = new double[nprocs];
    std::fill_n(tbound, nprocs, 0.0);
    tbound[this->myrank] = this->fs->rd->starts[0]*this->dz;
    MPI_Allreduce(
            tbound,
            this->lbound,
            nprocs,
            MPI_DOUBLE,
            MPI_SUM,
            this->comm);
    std::fill_n(tbound, nprocs, 0.0);
    tbound[this->myrank] = (this->fs->rd->starts[0] + this->fs->rd->subsizes[0])*this->dz;
    MPI_Allreduce(
            tbound,
            this->ubound,
            nprocs,
            MPI_DOUBLE,
            MPI_SUM,
            this->comm);
    delete[] tbound;
}

template <int particle_type, class rnumber, int interp_neighbours>
rFFTW_particles<particle_type, rnumber, interp_neighbours>::~rFFTW_particles()
{
    delete[] this->state;
    for (int i=0; i < this->integration_steps; i++)
    {
        delete[] this->rhs[i];
    }
    delete[] this->lbound;
    delete[] this->ubound;
}

template <int particle_type, class rnumber,  int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::sample_vec_field(
        rFFTW_interpolator<rnumber, interp_neighbours> *vec,
        double t,
        double *x,
        double *y,
        int *deriv)
{
    /* get grid coordinates */
    int *xg = new int[3*this->nparticles];
    double *xx = new double[3*this->nparticles];
    double *yy =  new double[3*this->nparticles];
    std::fill_n(yy, 3*this->nparticles, 0.0);
    vec->get_grid_coordinates(this->nparticles, this->ncomponents, x, xg, xx);
    /* perform interpolation */
    for (int p=0; p<this->nparticles; p++)
        (*vec)(t, xg + p*3, xx + p*3, yy + p*3, deriv);
    MPI_Allreduce(
            yy,
            y,
            3*this->nparticles,
            MPI_DOUBLE,
            MPI_SUM,
            this->comm);
    delete[] yy;
    delete[] xg;
    delete[] xx;
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::get_rhs(double t, double *x, double *y)
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->sample_vec_field(this->vel, t, x, y);
            break;
    }
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        std::copy(this->rhs[i],
                  this->rhs[i] + this->array_size,
                  this->rhs[i+1]);
}



template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::AdamsBashforth(int nsteps)
{
    ptrdiff_t ii;
    this->get_rhs(0, this->state, this->rhs[0]);
    switch(nsteps)
    {
        case 1:
            for (int p=0; p<this->nparticles; p++)
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*this->rhs[0][ii];
                }
            break;
        case 2:
            for (int p=0; p<this->nparticles; p++)
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(3*this->rhs[0][ii]
                                               -   this->rhs[1][ii])/2;
                }
            break;
        case 3:
            for (int p=0; p<this->nparticles; p++)
                for (int i=0; i<this->ncomponents; i++)
                {
                    ii = p*this->ncomponents+i;
                    this->state[ii] += this->dt*(23*this->rhs[0][ii]
                                               - 16*this->rhs[1][ii]
                                               +  5*this->rhs[2][ii])/12;
                }
            break;
        case 4:
            for (int p=0; p<this->nparticles; p++)
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
            for (int p=0; p<this->nparticles; p++)
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
            for (int p=0; p<this->nparticles; p++)
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


template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::step()
{
    this->AdamsBashforth((this->iteration < this->integration_steps) ?
                            this->iteration+1 :
                            this->integration_steps);
    this->iteration++;
}



template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::get_grid_coordinates(double *x, int *xg, double *xx)
{
    static double grid_size[] = {this->dx, this->dy, this->dz};
    double tval;
    std::fill_n(xg, this->nparticles*3, 0);
    std::fill_n(xx, this->nparticles*3, 0.0);
    for (int p=0; p<this->nparticles; p++)
    {
        for (int c=0; c<3; c++)
        {
            tval = floor(x[p*this->ncomponents+c]/grid_size[c]);
            xg[p*3+c] = MOD(int(tval), this->fs->rd->sizes[2-c]);
            xx[p*3+c] = (x[p*this->ncomponents+c] - tval*grid_size[c]) / grid_size[c];
        }
        /*xg[p*3+2] -= this->fs->rd->starts[0];
        if (this->myrank == this->fs->rd->rank[0] &&
            xg[p*3+2] > this->fs->rd->subsizes[0])
            xg[p*3+2] -= this->fs->rd->sizes[0];*/
    }
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::read(hid_t data_file_id)
{
    if (this->myrank == 0)
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
            this->comm);
    for (int i = 0; i<this->integration_steps; i++)
        MPI_Bcast(
                this->rhs[i],
                this->array_size,
                MPI_DOUBLE,
                0,
                this->comm);
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_particles<particle_type, rnumber, interp_neighbours>::write(hid_t data_file_id, bool write_rhs)
{
    if (this->myrank == 0)
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


/*****************************************************************************/
template class rFFTW_particles<VELOCITY_TRACER, float, 1>;
template class rFFTW_particles<VELOCITY_TRACER, float, 2>;
template class rFFTW_particles<VELOCITY_TRACER, float, 3>;
template class rFFTW_particles<VELOCITY_TRACER, float, 4>;
template class rFFTW_particles<VELOCITY_TRACER, float, 5>;
template class rFFTW_particles<VELOCITY_TRACER, float, 6>;
template class rFFTW_particles<VELOCITY_TRACER, double, 1>;
template class rFFTW_particles<VELOCITY_TRACER, double, 2>;
template class rFFTW_particles<VELOCITY_TRACER, double, 3>;
template class rFFTW_particles<VELOCITY_TRACER, double, 4>;
template class rFFTW_particles<VELOCITY_TRACER, double, 5>;
template class rFFTW_particles<VELOCITY_TRACER, double, 6>;
/*****************************************************************************/
