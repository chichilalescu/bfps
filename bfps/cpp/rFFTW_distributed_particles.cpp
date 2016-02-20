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



//#define NDEBUG

#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>

#include "base.hpp"
#include "rFFTW_distributed_particles.hpp"
#include "fftw_tools.hpp"


extern int myrank, nprocs;

template <int particle_type, class rnumber, int interp_neighbours>
rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::rFFTW_distributed_particles(
        const char *NAME,
        const hid_t data_file_id,
        rFFTW_interpolator<rnumber, interp_neighbours> *FIELD,
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

    this->interp_comm.resize(this->vel->descriptor->sizes[0]);
    this->interp_nprocs.resize(this->vel->descriptor->sizes[0]);
    int rmaxz, rminz;
    int color, key;
    for (int zg=0; zg<this->vel->descriptor->sizes[0]; zg++)
    {
        color = (this->vel->get_rank_info(
                    (zg+.5)*this->vel->dz, rminz, rmaxz) ? zg : MPI_UNDEFINED);
        key = zg - this->vel->descriptor->starts[0] + interp_neighbours;
        MPI_Comm_split(this->comm, color, key, &this->interp_comm[zg]);
        if (this->interp_comm[zg] != MPI_COMM_NULL)
            MPI_Comm_size(this->interp_comm[zg], &this->interp_nprocs[zg]);
        else
            this->interp_nprocs[zg] = 0;
    }
}

template <int particle_type, class rnumber, int interp_neighbours>
rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::~rFFTW_distributed_particles()
{
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        rFFTW_interpolator<rnumber, interp_neighbours> *field,
        const std::unordered_map<int, single_particle_state<particle_type>> &x,
        std::unordered_map<int, single_particle_state<POINT3D>> &y)
{
    double *yyy = new double[3];
    double *yy = new double[3];
    std::fill_n(yy, 3, 0);
    y.clear();
    int xg[3];
    double xx[3];
    for (int p=0; p<this->nparticles; p++)
    {
        auto pp = x.find(p);
        if (pp != x.end())
        {
            field->get_grid_coordinates(pp->second.data, xg, xx);
            (*field)(xg, xx, yy);
            if (this->interp_nprocs[xg[2]]>1)
            {
                DEBUG_MSG(
                        "iteration %d, zg is %d, nprocs is %d\n",
                        this->iteration, xg[2], this->interp_nprocs[xg[2]]);
                MPI_Allreduce(
                        yy,
                        yyy,
                        3,
                        MPI_DOUBLE,
                        MPI_SUM,
                        this->interp_comm[xg[2]]);
                y[p] = yyy;
            }
            else
                y[p] = yy;
        }
    }
    delete[] yy;
    delete[] yyy;
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::get_rhs(
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

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        rFFTW_interpolator<rnumber, interp_neighbours> *field,
        const char *dset_name)
{
    std::unordered_map<int, single_particle_state<POINT3D>> y;
    this->sample(field, this->state, y);
    this->write(dset_name, y);
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        rhs[i+1] = rhs[i];
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::redistribute(
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
    {
        int rminz, rmaxz;
        bool is_here = this->vel->get_rank_info(pp.second.data[2], rminz, rmaxz);
        //for (int i=0; i<2; i++)
        //{
        //    if (this->vel->get_rank() == nr[i])
        //        ps[i].push_back(pp.first);
        //}
    }
    /* prepare data for send recv */
    for (int i=0; i<2; i++)
        nps[i] = ps[i].size();
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (int i=0; i<2; i++)
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
    for (int i=0; i<2; i++)
        pr[i].resize(npr[i]);

    int buffer_size = (nps[0] > nps[1]) ? nps[0] : nps[1];
    buffer_size = (buffer_size > npr[0])? buffer_size : npr[0];
    buffer_size = (buffer_size > npr[1])? buffer_size : npr[1];
    //DEBUG_MSG("buffer size is %d\n", buffer_size);
    double *buffer = new double[buffer_size*this->ncomponents*(1+vals.size())];
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (int i=0; i<2; i++)
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
                              x[p].data + this->ncomponents,
                              buffer + pcounter*(1+vals.size())*this->ncomponents);
                    x.erase(p);
                    for (int tindex=0; tindex<vals.size(); tindex++)
                    {
                        std::copy(vals[tindex][p].data,
                                  vals[tindex][p].data + this->ncomponents,
                                  buffer + (pcounter*(1+vals.size()) + tindex+1)*this->ncomponents);
                        vals[tindex].erase(p);
                    }
                    pcounter++;
                }
                MPI_Send(
                        buffer,
                        nps[i]*(1+vals.size())*this->ncomponents,
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
                        npr[1-i]*(1+vals.size())*this->ncomponents,
                        MPI_DOUBLE,
                        rsrc,
                        2*(rsrc*this->nprocs + rdst)+1,
                        this->comm,
                        MPI_STATUS_IGNORE);
                int pcounter = 0;
                for (int p: pr[1-i])
                {
                    x[p] = buffer + (pcounter*(1+vals.size()))*this->ncomponents;
                    for (int tindex=0; tindex<vals.size(); tindex++)
                    {
                        vals[tindex][p] = buffer + (pcounter*(1+vals.size()) + tindex+1)*this->ncomponents;
                    }
                    pcounter++;
                }
            }
        }
    delete[] buffer;


#ifndef NDEBUG
    /* check that all particles at x are local */
    //for (auto &pp: x)
    //    if (this->vel->get_rank(pp.second.data[2]) != this->myrank)
    //    {
    //        DEBUG_MSG("found particle %d with rank %d\n",
    //                pp.first,
    //                this->vel->get_rank(pp.second.data[2]));
    //        assert(false);
    //    }
#endif
    //DEBUG_MSG("exiting redistribute\n");
}



template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::AdamsBashforth(
        const int nsteps)
{
    this->get_rhs(this->state, this->rhs[0]);
    //for (auto &pp: this->state)
    //    for (int i=0; i<this->ncomponents; i++)
    //        switch(nsteps)
    //        {
    //            case 1:
    //                pp.second[i] += this->dt*this->rhs[0][pp.first][i];
    //                break;
    //            case 2:
    //                pp.second[i] += this->dt*(3*this->rhs[0][pp.first][i]
    //                                        -   this->rhs[1][pp.first][i])/2;
    //                break;
    //            case 3:
    //                pp.second[i] += this->dt*(23*this->rhs[0][pp.first][i]
    //                                        - 16*this->rhs[1][pp.first][i]
    //                                        +  5*this->rhs[2][pp.first][i])/12;
    //                break;
    //            case 4:
    //                pp.second[i] += this->dt*(55*this->rhs[0][pp.first][i]
    //                                        - 59*this->rhs[1][pp.first][i]
    //                                        + 37*this->rhs[2][pp.first][i]
    //                                        -  9*this->rhs[3][pp.first][i])/24;
    //                break;
    //            case 5:
    //                pp.second[i] += this->dt*(1901*this->rhs[0][pp.first][i]
    //                                        - 2774*this->rhs[1][pp.first][i]
    //                                        + 2616*this->rhs[2][pp.first][i]
    //                                        - 1274*this->rhs[3][pp.first][i]
    //                                        +  251*this->rhs[4][pp.first][i])/720;
    //                break;
    //            case 6:
    //                pp.second[i] += this->dt*(4277*this->rhs[0][pp.first][i]
    //                                        - 7923*this->rhs[1][pp.first][i]
    //                                        + 9982*this->rhs[2][pp.first][i]
    //                                        - 7298*this->rhs[3][pp.first][i]
    //                                        + 2877*this->rhs[4][pp.first][i]
    //                                        -  475*this->rhs[5][pp.first][i])/1440;
    //                break;
    //        }
    //this->redistribute(this->state, this->rhs);
    this->roll_rhs();
}


template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::step()
{
    this->AdamsBashforth((this->iteration < this->integration_steps) ?
                          this->iteration+1 :
                          this->integration_steps);
    this->iteration++;
}


template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::read()
{
    double *temp = new double[this->chunk_size*this->ncomponents];
    int tmpint1, tmpint2;
    for (int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //read state
        if (this->myrank == 0)
            this->read_state_chunk(cindex, temp);
        MPI_Bcast(
                temp,
                this->chunk_size*this->ncomponents,
                MPI_DOUBLE,
                0,
                this->comm);
        for (int p=0; p<this->chunk_size; p++)
        {
            if (this->vel->get_rank_info(temp[this->ncomponents*p+2], tmpint1, tmpint2))
                this->state[p+cindex*this->chunk_size] = temp + this->ncomponents*p;
        }
        //read rhs
        if (this->iteration > 0)
            for (int i=0; i<this->integration_steps; i++)
            {
                if (this->myrank == 0)
                    this->read_rhs_chunk(cindex, i, temp);
                MPI_Bcast(
                        temp,
                        this->chunk_size*this->ncomponents,
                        MPI_DOUBLE,
                        0,
                        this->comm);
                for (int p=0; p<this->chunk_size; p++)
                {
                    auto pp = this->state.find(p+cindex*this->chunk_size);
                    if (pp != this->state.end())
                        this->rhs[i][p+cindex*this->chunk_size] = temp + this->ncomponents*p;
                }
            }
    }
    DEBUG_MSG("%s->state.size = %ld\n", this->name.c_str(), this->state.size());
    delete[] temp;
}

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const char *dset_name,
        std::unordered_map<int, single_particle_state<POINT3D>> &y)
{
    double *data = new double[this->nparticles*3];
    double *yy = new double[this->nparticles*3];
    int zmin_rank, zmax_rank;
    for (int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        std::fill_n(yy, this->chunk_size*3, 0);
        for (int p=0; p<this->chunk_size; p++)
        {
            auto pp = y.find(p+cindex*this->chunk_size);
            if (pp != y.end())
            {
                this->vel->get_rank_info(pp->second.data[2], zmax_rank, zmin_rank);
                if (this->myrank == zmin_rank)
                    std::copy(pp->second.data,
                              pp->second.data + 3,
                              yy + pp->first*3);
            }
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

template <int particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const bool write_rhs)
{
    double *temp0 = new double[this->chunk_size*this->ncomponents];
    double *temp1 = new double[this->chunk_size*this->ncomponents];
    int zmin_rank, zmax_rank;
    for (int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //write state
        std::fill_n(temp0, this->ncomponents*this->chunk_size, 0);
        for (int p=0; p<this->chunk_size; p++)
        {
            auto pp = this->state.find(p + cindex*this->chunk_size);
            if (pp != this->state.end())
            {
                this->vel->get_rank_info(pp->second.data[2], zmax_rank, zmin_rank);
                if (this->myrank == zmin_rank)
                    std::copy(pp->second.data,
                              pp->second.data + this->ncomponents,
                              temp0 + pp->first*this->ncomponents);
            }
        }
        MPI_Allreduce(
                temp0,
                temp1,
                this->ncomponents*this->chunk_size,
                MPI_DOUBLE,
                MPI_SUM,
                this->comm);
        if (this->myrank == 0)
            this->write_state_chunk(cindex, temp1);
        //write rhs
        if (write_rhs)
            for (int i=0; i<this->integration_steps; i++)
            {
                std::fill_n(temp0, this->ncomponents*this->chunk_size, 0);
                for (int p=0; p<this->chunk_size; p++)
                {
                    auto pp = this->rhs[i].find(p + cindex*this->chunk_size);
                    if (pp != this->rhs[i].end())
                    {
                        this->vel->get_rank_info(pp->second.data[2], zmax_rank, zmin_rank);
                        if (this->myrank == zmin_rank)
                            std::copy(pp->second.data,
                                      pp->second.data + this->ncomponents,
                                      temp0 + pp->first*this->ncomponents);
                    }
                }
                MPI_Allreduce(
                        temp0,
                        temp1,
                        this->ncomponents*this->chunk_size,
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
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 1>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 2>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 3>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 4>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 5>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, float, 6>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 1>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 2>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 3>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 4>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 5>;
template class rFFTW_distributed_particles<VELOCITY_TRACER, double, 6>;
/*****************************************************************************/

