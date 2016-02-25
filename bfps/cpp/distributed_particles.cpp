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
#include "distributed_particles.hpp"
#include "fftw_tools.hpp"


extern int myrank, nprocs;

template <particle_types particle_type, class rnumber, int interp_neighbours>
distributed_particles<particle_type, rnumber, interp_neighbours>::distributed_particles(
        const char *NAME,
        const hid_t data_file_id,
        interpolator<rnumber, interp_neighbours> *FIELD,
        const int TRAJ_SKIP,
        const int INTEGRATION_STEPS) : particles_io_base<particle_type>(
            NAME,
            TRAJ_SKIP,
            data_file_id,
            FIELD->descriptor->comm)
{
    assert((INTEGRATION_STEPS <= 6) &&
           (INTEGRATION_STEPS >= 1));
    this->vel = FIELD;
    this->rhs.resize(INTEGRATION_STEPS);
    this->integration_steps = INTEGRATION_STEPS;
    this->state.reserve(2*this->nparticles / this->nprocs);
    for (unsigned int i=0; i<this->rhs.size(); i++)
        this->rhs[i].reserve(2*this->nparticles / this->nprocs);
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
distributed_particles<particle_type, rnumber, interp_neighbours>::~distributed_particles()
{
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        interpolator<rnumber, interp_neighbours> *field,
        const std::unordered_map<int, single_particle_state<particle_type>> &x,
        std::unordered_map<int, single_particle_state<POINT3D>> &y)
{
    double *yy = new double[3];
    y.clear();
    for (auto &pp: x)
    {
        (*field)(pp.second.data, yy);
        y[pp.first] = yy;
    }
    delete[] yy;
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::get_rhs(
        const std::unordered_map<int, single_particle_state<particle_type>> &x,
        std::unordered_map<int, single_particle_state<particle_type>> &y)
{
    std::unordered_map<int, single_particle_state<POINT3D>> yy;
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->sample(this->vel, this->state, yy);
            y.clear();
            for (auto &pp: x)
                y[pp.first] = yy[pp.first].data;
            break;
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        interpolator<rnumber, interp_neighbours> *field,
        const char *dset_name)
{
    std::unordered_map<int, single_particle_state<POINT3D>> y;
    this->sample(field, this->state, y);
    this->write(dset_name, y);
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        rhs[i+1] = rhs[i];
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::redistribute(
        std::unordered_map<int, single_particle_state<particle_type>> &x,
        std::vector<std::unordered_map<int, single_particle_state<particle_type>>> &vals)
{
    //DEBUG_MSG("entered redistribute\n");
    /* neighbouring rank offsets */
    int ro[2];
    ro[0] = -1;
    ro[1] = 1;
    /* neighbouring ranks */
    int nr[2];
    nr[0] = MOD(this->myrank+ro[0], this->nprocs);
    nr[1] = MOD(this->myrank+ro[1], this->nprocs);
    /* particles to send, particles to receive */
    std::vector<int> ps[2], pr[2];
    /* number of particles to send, number of particles to receive */
    int nps[2], npr[2];
    int rsrc, rdst;
    /* get list of id-s to send */
    for (auto &pp: x)
        for (unsigned int i=0; i<2; i++)
            if (this->vel->get_rank(pp.second.data[2]) == nr[i])
                ps[i].push_back(pp.first);
    /* prepare data for send recv */
    for (unsigned int i=0; i<2; i++)
        nps[i] = ps[i].size();
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (unsigned int i=0; i<2; i++)
        {
            rdst = MOD(rsrc+ro[i], this->nprocs);
            if (this->myrank == rsrc)
                MPI_Send(
                        nps+i,
                        1,
                        MPI_INTEGER,
                        rdst,
                        2*(rsrc*this->nprocs + rdst)+i,
                        this->comm);
            if (this->myrank == rdst)
                MPI_Recv(
                        npr+1-i,
                        1,
                        MPI_INTEGER,
                        rsrc,
                        2*(rsrc*this->nprocs + rdst)+i,
                        this->comm,
                        MPI_STATUS_IGNORE);
        }
    //DEBUG_MSG("I have to send %d %d particles\n", nps[0], nps[1]);
    //DEBUG_MSG("I have to recv %d %d particles\n", npr[0], npr[1]);
    for (unsigned int i=0; i<2; i++)
        pr[i].resize(npr[i]);

    int buffer_size = (nps[0] > nps[1]) ? nps[0] : nps[1];
    buffer_size = (buffer_size > npr[0])? buffer_size : npr[0];
    buffer_size = (buffer_size > npr[1])? buffer_size : npr[1];
    //DEBUG_MSG("buffer size is %d\n", buffer_size);
    double *buffer = new double[buffer_size*state_dimension(particle_type)*(1+vals.size())];
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (unsigned int i=0; i<2; i++)
        {
            rdst = MOD(rsrc+ro[i], this->nprocs);
            if (this->myrank == rsrc && nps[i] > 0)
            {
                MPI_Send(
                        &ps[i].front(),
                        nps[i],
                        MPI_INTEGER,
                        rdst,
                        2*(rsrc*this->nprocs + rdst),
                        this->comm);
                int pcounter = 0;
                for (int p: ps[i])
                {
                    std::copy(x[p].data,
                              x[p].data + state_dimension(particle_type),
                              buffer + pcounter*(1+vals.size())*state_dimension(particle_type));
                    x.erase(p);
                    for (unsigned int tindex=0; tindex<vals.size(); tindex++)
                    {
                        std::copy(vals[tindex][p].data,
                                  vals[tindex][p].data + state_dimension(particle_type),
                                  buffer + (pcounter*(1+vals.size()) + tindex+1)*state_dimension(particle_type));
                        vals[tindex].erase(p);
                    }
                    pcounter++;
                }
                MPI_Send(
                        buffer,
                        nps[i]*(1+vals.size())*state_dimension(particle_type),
                        MPI_DOUBLE,
                        rdst,
                        2*(rsrc*this->nprocs + rdst)+1,
                        this->comm);
            }
            if (this->myrank == rdst && npr[1-i] > 0)
            {
                MPI_Recv(
                        &pr[1-i].front(),
                        npr[1-i],
                        MPI_INTEGER,
                        rsrc,
                        2*(rsrc*this->nprocs + rdst),
                        this->comm,
                        MPI_STATUS_IGNORE);
                MPI_Recv(
                        buffer,
                        npr[1-i]*(1+vals.size())*state_dimension(particle_type),
                        MPI_DOUBLE,
                        rsrc,
                        2*(rsrc*this->nprocs + rdst)+1,
                        this->comm,
                        MPI_STATUS_IGNORE);
                unsigned int pcounter = 0;
                for (int p: pr[1-i])
                {
                    x[p] = buffer + (pcounter*(1+vals.size()))*state_dimension(particle_type);
                    for (unsigned int tindex=0; tindex<vals.size(); tindex++)
                    {
                        vals[tindex][p] = buffer + (pcounter*(1+vals.size()) + tindex+1)*state_dimension(particle_type);
                    }
                    pcounter++;
                }
            }
        }
    delete[] buffer;


#ifndef NDEBUG
    /* check that all particles at x are local */
    for (auto &pp: x)
        if (this->vel->get_rank(pp.second.data[2]) != this->myrank)
        {
            DEBUG_MSG("found particle %d with rank %d\n",
                    pp.first,
                    this->vel->get_rank(pp.second.data[2]));
            assert(false);
        }
#endif
    //DEBUG_MSG("exiting redistribute\n");
}



template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::AdamsBashforth(
        const int nsteps)
{
    this->get_rhs(this->state, this->rhs[0]);
    for (auto &pp: this->state)
        for (unsigned int i=0; i<state_dimension(particle_type); i++)
            switch(nsteps)
            {
                case 1:
                    pp.second[i] += this->dt*this->rhs[0][pp.first][i];
                    break;
                case 2:
                    pp.second[i] += this->dt*(3*this->rhs[0][pp.first][i]
                                            -   this->rhs[1][pp.first][i])/2;
                    break;
                case 3:
                    pp.second[i] += this->dt*(23*this->rhs[0][pp.first][i]
                                            - 16*this->rhs[1][pp.first][i]
                                            +  5*this->rhs[2][pp.first][i])/12;
                    break;
                case 4:
                    pp.second[i] += this->dt*(55*this->rhs[0][pp.first][i]
                                            - 59*this->rhs[1][pp.first][i]
                                            + 37*this->rhs[2][pp.first][i]
                                            -  9*this->rhs[3][pp.first][i])/24;
                    break;
                case 5:
                    pp.second[i] += this->dt*(1901*this->rhs[0][pp.first][i]
                                            - 2774*this->rhs[1][pp.first][i]
                                            + 2616*this->rhs[2][pp.first][i]
                                            - 1274*this->rhs[3][pp.first][i]
                                            +  251*this->rhs[4][pp.first][i])/720;
                    break;
                case 6:
                    pp.second[i] += this->dt*(4277*this->rhs[0][pp.first][i]
                                            - 7923*this->rhs[1][pp.first][i]
                                            + 9982*this->rhs[2][pp.first][i]
                                            - 7298*this->rhs[3][pp.first][i]
                                            + 2877*this->rhs[4][pp.first][i]
                                            -  475*this->rhs[5][pp.first][i])/1440;
                    break;
            }
    this->redistribute(this->state, this->rhs);
    this->roll_rhs();
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::step()
{
    this->AdamsBashforth((this->iteration < this->integration_steps) ?
                            this->iteration+1 :
                            this->integration_steps);
    this->iteration++;
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::read()
{
    double *temp = new double[this->chunk_size*state_dimension(particle_type)];
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //read state
        if (this->myrank == 0)
            this->read_state_chunk(cindex, temp);
        MPI_Bcast(
                temp,
                this->chunk_size*state_dimension(particle_type),
                MPI_DOUBLE,
                0,
                this->comm);
        for (unsigned int p=0; p<this->chunk_size; p++)
        {
            if (this->vel->get_rank(temp[state_dimension(particle_type)*p+2]) == this->myrank)
                this->state[p+cindex*this->chunk_size] = temp + state_dimension(particle_type)*p;
        }
        //read rhs
        if (this->iteration > 0)
            for (int i=0; i<this->integration_steps; i++)
            {
                if (this->myrank == 0)
                    this->read_rhs_chunk(cindex, i, temp);
                MPI_Bcast(
                        temp,
                        this->chunk_size*state_dimension(particle_type),
                        MPI_DOUBLE,
                        0,
                        this->comm);
                for (unsigned int p=0; p<this->chunk_size; p++)
                {
                    auto pp = this->state.find(p+cindex*this->chunk_size);
                    if (pp != this->state.end())
                        this->rhs[i][p+cindex*this->chunk_size] = temp + state_dimension(particle_type)*p;
                }
            }
    }
    DEBUG_MSG("%s->state.size = %ld\n", this->name.c_str(), this->state.size());
    delete[] temp;
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const char *dset_name,
        std::unordered_map<int, single_particle_state<POINT3D>> &y)
{
    double *data = new double[this->nparticles*3];
    double *yy = new double[this->nparticles*3];
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        std::fill_n(yy, this->chunk_size*3, 0);
        for (unsigned int p=0; p<this->chunk_size; p++)
        {
            auto pp = y.find(p+cindex*this->chunk_size);
            if (pp != y.end())
                std::copy(pp->second.data,
                          pp->second.data + 3,
                          yy + pp->first*3);
        }
        MPI_Allreduce(
                yy,
                data,
                3*this->nparticles,
                MPI_DOUBLE,
                MPI_SUM,
                this->comm);
        if (this->myrank == 0)
            this->write_point3D_chunk(dset_name, cindex, data);
    }
    delete[] yy;
    delete[] data;
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const bool write_rhs)
{
    double *temp0 = new double[this->chunk_size*state_dimension(particle_type)];
    double *temp1 = new double[this->chunk_size*state_dimension(particle_type)];
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //write state
        std::fill_n(temp0, state_dimension(particle_type)*this->chunk_size, 0);
        for (unsigned int p=0; p<this->chunk_size; p++)
        {
            auto pp = this->state.find(p + cindex*this->chunk_size);
            if (pp != this->state.end())
                std::copy(pp->second.data,
                          pp->second.data + state_dimension(particle_type),
                          temp0 + pp->first*state_dimension(particle_type));
        }
        MPI_Allreduce(
                temp0,
                temp1,
                state_dimension(particle_type)*this->chunk_size,
                MPI_DOUBLE,
                MPI_SUM,
                this->comm);
        if (this->myrank == 0)
            this->write_state_chunk(cindex, temp1);
        //write rhs
        if (write_rhs)
            for (int i=0; i<this->integration_steps; i++)
            {
                std::fill_n(temp0, state_dimension(particle_type)*this->chunk_size, 0);
                for (unsigned int p=0; p<this->chunk_size; p++)
                {
                    auto pp = this->rhs[i].find(p + cindex*this->chunk_size);
                    if (pp != this->rhs[i].end())
                        std::copy(pp->second.data,
                                  pp->second.data + state_dimension(particle_type),
                                  temp0 + pp->first*state_dimension(particle_type));
                }
                MPI_Allreduce(
                        temp0,
                        temp1,
                        state_dimension(particle_type)*this->chunk_size,
                        MPI_DOUBLE,
                        MPI_SUM,
                        this->comm);
                if (this->myrank == 0)
                    this->write_rhs_chunk(cindex, i, temp1);
            }
    }
    delete[] temp0;
    delete[] temp1;
}


/*****************************************************************************/
template class distributed_particles<VELOCITY_TRACER, float, 1>;
template class distributed_particles<VELOCITY_TRACER, float, 2>;
template class distributed_particles<VELOCITY_TRACER, float, 3>;
template class distributed_particles<VELOCITY_TRACER, float, 4>;
template class distributed_particles<VELOCITY_TRACER, float, 5>;
template class distributed_particles<VELOCITY_TRACER, float, 6>;
template class distributed_particles<VELOCITY_TRACER, double, 1>;
template class distributed_particles<VELOCITY_TRACER, double, 2>;
template class distributed_particles<VELOCITY_TRACER, double, 3>;
template class distributed_particles<VELOCITY_TRACER, double, 4>;
template class distributed_particles<VELOCITY_TRACER, double, 5>;
template class distributed_particles<VELOCITY_TRACER, double, 6>;
/*****************************************************************************/
