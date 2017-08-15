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
#include "particles.hpp"
#include "fftw_tools.hpp"


extern int myrank, nprocs;

template <particle_types particle_type, class rnumber, int interp_neighbours>
particles<particle_type, rnumber, interp_neighbours>::particles(
        const char *NAME,
        const hid_t data_file_id,
        interpolator_base<rnumber, interp_neighbours> *VEL,
        const int TRAJ_SKIP,
        const int INTEGRATION_STEPS) : particles_io_base<particle_type>(
            NAME,
            TRAJ_SKIP,
            data_file_id,
            VEL->descriptor->comm)
{
    assert((INTEGRATION_STEPS <= 6) &&
           (INTEGRATION_STEPS >= 1));
    this->vel = VEL;
    this->integration_steps = INTEGRATION_STEPS;
    this->array_size = this->nparticles * state_dimension(particle_type);
    this->state = new double[this->array_size];
    std::fill_n(this->state, this->array_size, 0.0);
    for (int i=0; i < this->integration_steps; i++)
    {
        this->rhs[i] = new double[this->array_size];
        std::fill_n(this->rhs[i], this->array_size, 0.0);
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
particles<particle_type, rnumber, interp_neighbours>::~particles()
{
    delete[] this->state;
    for (int i=0; i < this->integration_steps; i++)
    {
        delete[] this->rhs[i];
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::get_rhs(double *x, double *y)
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->vel->sample(this->nparticles, state_dimension(particle_type), x, y);
            break;
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        std::copy(this->rhs[i],
                  this->rhs[i] + this->array_size,
                  this->rhs[i+1]);
}



template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::AdamsBashforth(
        const int nsteps)
{
    ptrdiff_t ii;
    this->get_rhs(this->state, this->rhs[0]);
    switch(nsteps)
    {
        case 1:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
                    this->state[ii] += this->dt*this->rhs[0][ii];
                }
            break;
        case 2:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
                    this->state[ii] += this->dt*(3*this->rhs[0][ii]
                                               -   this->rhs[1][ii])/2;
                }
            break;
        case 3:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
                    this->state[ii] += this->dt*(23*this->rhs[0][ii]
                                               - 16*this->rhs[1][ii]
                                               +  5*this->rhs[2][ii])/12;
                }
            break;
        case 4:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
                    this->state[ii] += this->dt*(55*this->rhs[0][ii]
                                               - 59*this->rhs[1][ii]
                                               + 37*this->rhs[2][ii]
                                               -  9*this->rhs[3][ii])/24;
                }
            break;
        case 5:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
                    this->state[ii] += this->dt*(1901*this->rhs[0][ii]
                                               - 2774*this->rhs[1][ii]
                                               + 2616*this->rhs[2][ii]
                                               - 1274*this->rhs[3][ii]
                                               +  251*this->rhs[4][ii])/720;
                }
            break;
        case 6:
            for (unsigned int p=0; p<this->nparticles; p++)
                for (unsigned int i=0; i<state_dimension(particle_type); i++)
                {
                    ii = p*state_dimension(particle_type)+i;
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


template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::step()
{
    this->AdamsBashforth((this->iteration < this->integration_steps) ?
                            this->iteration+1 :
                            this->integration_steps);
    this->iteration++;
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::read()
{
    if (this->myrank == 0)
        for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
        {
            this->read_state_chunk(cindex, this->state+cindex*this->chunk_size*state_dimension(particle_type));
            if (this->iteration > 0)
                for (int i=0; i<this->integration_steps; i++)
                    this->read_rhs_chunk(cindex, i, this->rhs[i]+cindex*this->chunk_size*state_dimension(particle_type));
        }
    MPI_Bcast(
            this->state,
            this->array_size,
            MPI_DOUBLE,
            0,
            this->comm);
    if (this->iteration > 0)
        for (int i = 0; i<this->integration_steps; i++)
            MPI_Bcast(
                    this->rhs[i],
                    this->array_size,
                    MPI_DOUBLE,
                    0,
                    this->comm);
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::write(
        const bool write_rhs)
{
    if (this->myrank == 0)
        for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
        {
            this->write_state_chunk(cindex, this->state+cindex*this->chunk_size*state_dimension(particle_type));
            if (write_rhs)
                for (int i=0; i<this->integration_steps; i++)
                    this->write_rhs_chunk(cindex, i, this->rhs[i]+cindex*this->chunk_size*state_dimension(particle_type));
        }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void particles<particle_type, rnumber, interp_neighbours>::sample(
        interpolator_base<rnumber, interp_neighbours> *field,
        const char *dset_name)
{
    double *y = new double[this->nparticles*3];
    field->sample(this->nparticles, state_dimension(particle_type), this->state, y);
    if (this->myrank == 0)
        for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
            this->write_point3D_chunk(dset_name, cindex, y+cindex*this->chunk_size*3);
    delete[] y;
}


/*****************************************************************************/
template class particles<VELOCITY_TRACER, float, 1>;
template class particles<VELOCITY_TRACER, float, 2>;
template class particles<VELOCITY_TRACER, float, 3>;
template class particles<VELOCITY_TRACER, float, 4>;
template class particles<VELOCITY_TRACER, float, 5>;
template class particles<VELOCITY_TRACER, float, 6>;
template class particles<VELOCITY_TRACER, double, 1>;
template class particles<VELOCITY_TRACER, double, 2>;
template class particles<VELOCITY_TRACER, double, 3>;
template class particles<VELOCITY_TRACER, double, 4>;
template class particles<VELOCITY_TRACER, double, 5>;
template class particles<VELOCITY_TRACER, double, 6>;
/*****************************************************************************/
