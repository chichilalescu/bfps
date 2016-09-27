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
#include <set>
#include <algorithm>
#include <ctime>

#include "base.hpp"
#include "rFFTW_distributed_particles.hpp"
#include "fftw_tools.hpp"
#include "scope_timer.hpp"


extern int myrank, nprocs;

template <particle_types particle_type, class rnumber, int interp_neighbours>
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
    TIMEZONE("rFFTW_distributed_particles::rFFTW_distributed_particles");
    /* check that integration_steps has a valid value.
     * If NDEBUG is defined, "assert" doesn't do anything.
     * With NDEBUG defined, and an invalid INTEGRATION_STEPS,
     * the particles will simply sit still.
     * */
    assert((INTEGRATION_STEPS <= 6) &&
           (INTEGRATION_STEPS >= 1));
    /* check that the field layout is compatible with this class.
     * if it's not, the code will fail in bad ways, most likely ending up
     * with various CPUs locked in some MPI send/receive.
     * therefore I prefer to just kill the code at this point,
     * no matter whether or not NDEBUG is present.
     * */
    if (interp_neighbours*2+2 > FIELD->descriptor->subsizes[0])
    {
        DEBUG_MSG("parameters incompatible with rFFTW_distributed_particles.\n"
                  "interp kernel size is %d, local_z_size is %d\n",
                  interp_neighbours*2+2, FIELD->descriptor->subsizes[0]);
        if (FIELD->descriptor->myrank == 0)
            std::cerr << "parameters incompatible with rFFTW_distributed_particles." << std::endl;
        exit(0);
    }
    this->vel = FIELD;
    this->rhs.resize(INTEGRATION_STEPS);
    this->integration_steps = INTEGRATION_STEPS;
    //this->state.reserve(2*this->nparticles / this->nprocs);
    //for (unsigned int i=0; i<this->rhs.size(); i++)
    //    this->rhs[i].reserve(2*this->nparticles / this->nprocs);

    /* build communicators and stuff for interpolation */

    /* number of processors per domain */
    this->domain_nprocs[-1] = 2; // domain in common with lower z CPU
    this->domain_nprocs[ 0] = 1; // local domain
    this->domain_nprocs[ 1] = 2; // domain in common with higher z CPU

    /* initialize domain bins */
    this->domain_particles[-1] = std::set<int>();
    this->domain_particles[ 0] = std::set<int>();
    this->domain_particles[ 1] = std::set<int>();
    //this->domain_particles[-1].reserve(unsigned(
    //            1.5*(interp_neighbours*2+2)*
    //            float(this->nparticles) /
    //            this->nprocs));
    //this->domain_particles[ 1].reserve(unsigned(
    //            1.5*(interp_neighbours*2+2)*
    //            float(this->nparticles) /
    //            this->nprocs));
    //this->domain_particles[ 0].reserve(unsigned(
    //            1.5*(this->vel->descriptor->subsizes[0] - interp_neighbours*2-2)*
    //            float(this->nparticles) /
    //            this->nprocs));

    int color, key;
    MPI_Comm tmpcomm;
    for (int rank=0; rank<this->nprocs; rank++)
    {
        color = MPI_UNDEFINED;
        key = MPI_UNDEFINED;
        if (this->myrank == rank)
        {
            color = rank;
            key = 0;
        }
        if (this->myrank == MOD(rank + 1, this->nprocs))
        {
            color = rank;
            key = 1;
        }
        MPI_Comm_split(this->comm, color, key, &tmpcomm);
        if (this->myrank == rank)
            this->domain_comm[ 1] = tmpcomm;
        if (this->myrank == MOD(rank+1, this->nprocs))
            this->domain_comm[-1] = tmpcomm;

    }

    /* following code may be useful in the future for the general case */
    //this->interp_comm.resize(this->vel->descriptor->sizes[0]);
    //this->interp_nprocs.resize(this->vel->descriptor->sizes[0]);
    //for (int zg=0; zg<this->vel->descriptor->sizes[0]; zg++)
    //{
    //    color = (this->vel->get_rank_info(
    //                (zg+.5)*this->vel->dz, rminz, rmaxz) ? zg : MPI_UNDEFINED);
    //    key = zg - this->vel->descriptor->starts[0] + interp_neighbours;
    //    MPI_Comm_split(this->comm, color, key, &this->interp_comm[zg]);
    //    if (this->interp_comm[zg] != MPI_COMM_NULL)
    //        MPI_Comm_size(this->interp_comm[zg], &this->interp_nprocs[zg]);
    //    else
    //        this->interp_nprocs[zg] = 0;
    //}
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::~rFFTW_distributed_particles()
{
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        rFFTW_interpolator<rnumber, interp_neighbours> *field,
        const std::map<int, single_particle_state<particle_type>> &x,
        const std::map<int, std::set<int>> &dp,
        std::map<int, single_particle_state<POINT3D>> &y)
{
    TIMEZONE("rFFTW_distributed_particles::sample");
    double *yyy;
    double *yy;
    y.clear();
    /* local z domain */
    yy = new double[3];
    for (auto p: dp.at(0))
    {
        (*field)(x.find(p)->second.data, yy);
        y[p] = yy;
    }
    delete[] yy;
    /* boundary z domains */
    int domain_index;
    for (int rankpair = 0; rankpair < this->nprocs; rankpair++)
    {
        if (this->myrank == rankpair)
            domain_index = 1;
        if (this->myrank == MOD(rankpair+1, this->nprocs))
            domain_index = -1;
        if (this->myrank == rankpair ||
            this->myrank == MOD(rankpair+1, this->nprocs))
        {
            yy = new double[3*dp.at(domain_index).size()];
            yyy = new double[3*dp.at(domain_index).size()];
            int tindex;
            tindex = 0;
            // can this sorting be done more efficiently?
            std::vector<int> ordered_dp;
            ordered_dp.reserve(dp.at(domain_index).size());
            for (auto p: dp.at(domain_index))
                ordered_dp.push_back(p);
            //std::set<int> ordered_dp(dp.at(domain_index));
            std::sort(ordered_dp.begin(), ordered_dp.end());

            for (auto p: ordered_dp)
            //for (auto p: dp.at(domain_index))
            {
                (*field)(x.at(p).data, yy + tindex*3);
                tindex++;
            }
            MPI_Allreduce(
                    yy,
                    yyy,
                    3*dp.at(domain_index).size(),
                    MPI_DOUBLE,
                    MPI_SUM,
                    this->domain_comm[domain_index]);
            tindex = 0;
            for (auto p: ordered_dp)
            //for (auto p: dp.at(domain_index))
            {
                y[p] = yyy + tindex*3;
                tindex++;
            }
            delete[] yy;
            delete[] yyy;
        }
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::get_rhs(
        const std::map<int, single_particle_state<particle_type>> &x,
        const std::map<int, std::set<int>> &dp,
        std::map<int, single_particle_state<particle_type>> &y)
{
    std::map<int, single_particle_state<POINT3D>> yy;
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->sample(this->vel, x, dp, yy);
            y.clear();
            //y.reserve(yy.size());
            //y.rehash(this->nparticles);
            for (auto &pp: yy)
                y[pp.first] = pp.second.data;
            break;
    }
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::sample(
        rFFTW_interpolator<rnumber, interp_neighbours> *field,
        const char *dset_name)
{
    std::map<int, single_particle_state<POINT3D>> y;
    this->sample(field, this->state, this->domain_particles, y);
    this->write(dset_name, y);
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::roll_rhs()
{
    for (int i=this->integration_steps-2; i>=0; i--)
        rhs[i+1] = rhs[i];
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::redistribute(
        std::map<int, single_particle_state<particle_type>> &x,
        std::vector<std::map<int, single_particle_state<particle_type>>> &vals,
        std::map<int, std::set<int>> &dp)
{
    TIMEZONE("rFFTW_distributed_particles::redistribute");
    //DEBUG_MSG("entered redistribute\n");
    /* get new distribution of particles */
    std::map<int, std::set<int>> newdp;
    {
        TIMEZONE("sort_into_domains");
        this->sort_into_domains(x, newdp);
    }
    /* take care of particles that are leaving the shared domains */
    int dindex[2] = {-1, 1};
    // for each D of the 2 shared domains
    {
        TIMEZONE("Loop1");
        for (int di=0; di<2; di++)
            // for all particles previously in D
            for (auto p: dp[dindex[di]])
            {
                // if the particle is no longer in D
                if (newdp[dindex[di]].find(p) == newdp[dindex[di]].end())
                {
                    // and the particle is not in the local domain
                    if (newdp[0].find(p) == newdp[0].end())
                    {
                        // remove the particle from the local list
                        x.erase(p);
                        for (unsigned int i=0; i<vals.size(); i++)
                            vals[i].erase(p);
                    }
                    // if the particle is in the local domain, do nothing
                }
            }
    }
    /* take care of particles that are entering the shared domains */
    /* neighbouring rank offsets */
    int ro[2];
    ro[0] = -1;
    ro[1] = 1;
    /* particles to send, particles to receive */
    std::vector<int> ps[2], pr[2];
    for (int tcounter = 0; tcounter < 2; tcounter++)
    {
        ps[tcounter].reserve(newdp[dindex[tcounter]].size());
    }
    /* number of particles to send, number of particles to receive */
    int nps[2], npr[2];
    int rsrc, rdst;
    /* get list of id-s to send */
    {
        TIMEZONE("Loop2");
        for (auto &p: dp[0])
        {
            for (int di=0; di<2; di++)
            {
                if (newdp[dindex[di]].find(p) != newdp[dindex[di]].end())
                    ps[di].push_back(p);
            }
        }
    }
    /* prepare data for send recv */
    for (int i=0; i<2; i++)
        nps[i] = ps[i].size();
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (int i=0; i<2; i++)
        {
            rdst = MOD(rsrc+ro[i], this->nprocs);
            if (this->myrank == rsrc){
                TIMEZONE("MPI_Send");
                MPI_Send(
                        nps+i,
                        1,
                        MPI_INTEGER,
                        rdst,
                        2*(rsrc*this->nprocs + rdst)+i,
                        this->comm);
            }
            if (this->myrank == rdst){
                TIMEZONE("MPI_Recv");
                MPI_Recv(
                        npr+1-i,
                        1,
                        MPI_INTEGER,
                        rsrc,
                        2*(rsrc*this->nprocs + rdst)+i,
                        this->comm,
                        MPI_STATUS_IGNORE);
            }
        }
    //DEBUG_MSG("I have to send %d %d particles\n", nps[0], nps[1]);
    //DEBUG_MSG("I have to recv %d %d particles\n", npr[0], npr[1]);
    for (int i=0; i<2; i++)
        pr[i].resize(npr[i]);

    int buffer_size = (nps[0] > nps[1]) ? nps[0] : nps[1];
    buffer_size = (buffer_size > npr[0])? buffer_size : npr[0];
    buffer_size = (buffer_size > npr[1])? buffer_size : npr[1];
    //DEBUG_MSG("buffer size is %d\n", buffer_size);
    double *buffer = new double[buffer_size*state_dimension(particle_type)*(1+vals.size())];
    for (rsrc = 0; rsrc<this->nprocs; rsrc++)
        for (int i=0; i<2; i++)
        {
            rdst = MOD(rsrc+ro[i], this->nprocs);
            if (this->myrank == rsrc && nps[i] > 0)
            {
                TIMEZONE("this->myrank == rsrc && nps[i] > 0");
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
                    for (unsigned int tindex=0; tindex<vals.size(); tindex++)
                    {
                        std::copy(vals[tindex][p].data,
                                  vals[tindex][p].data + state_dimension(particle_type),
                                  buffer + (pcounter*(1+vals.size()) + tindex+1)*state_dimension(particle_type));
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
                TIMEZONE("this->myrank == rdst && npr[1-i] > 0");
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
                int pcounter = 0;
                for (int p: pr[1-i])
                {
                    x[p] = buffer + (pcounter*(1+vals.size()))*state_dimension(particle_type);
                    newdp[1-i].insert(p);
                    for (unsigned int tindex=0; tindex<vals.size(); tindex++)
                    {
                        vals[tindex][p] = buffer + (pcounter*(1+vals.size()) + tindex+1)*state_dimension(particle_type);
                    }
                    pcounter++;
                }
            }
        }
    delete[] buffer;
    // x has been changed, so newdp is obsolete
    // we need to sort into domains again
    {
        TIMEZONE("sort_into_domains2");
        this->sort_into_domains(x, dp);
    }

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



template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::AdamsBashforth(
        const int nsteps)
{
    this->get_rhs(this->state, this->domain_particles, this->rhs[0]);
#define AdamsBashforth_LOOP_PREAMBLE \
    for (auto &pp: this->state) \
        for (unsigned int i=0; i<state_dimension(particle_type); i++)
    switch(nsteps)
    {
        case 1:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*this->rhs[0][pp.first][i];
            break;
        case 2:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*(3*this->rhs[0][pp.first][i]
                                    -   this->rhs[1][pp.first][i])/2;
            break;
        case 3:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*(23*this->rhs[0][pp.first][i]
                                    - 16*this->rhs[1][pp.first][i]
                                    +  5*this->rhs[2][pp.first][i])/12;
            break;
        case 4:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*(55*this->rhs[0][pp.first][i]
                                    - 59*this->rhs[1][pp.first][i]
                                    + 37*this->rhs[2][pp.first][i]
                                    -  9*this->rhs[3][pp.first][i])/24;
            break;
        case 5:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*(1901*this->rhs[0][pp.first][i]
                                    - 2774*this->rhs[1][pp.first][i]
                                    + 2616*this->rhs[2][pp.first][i]
                                    - 1274*this->rhs[3][pp.first][i]
                                    +  251*this->rhs[4][pp.first][i])/720;
            break;
        case 6:
            AdamsBashforth_LOOP_PREAMBLE
            pp.second[i] += this->dt*(4277*this->rhs[0][pp.first][i]
                                    - 7923*this->rhs[1][pp.first][i]
                                    + 9982*this->rhs[2][pp.first][i]
                                    - 7298*this->rhs[3][pp.first][i]
                                    + 2877*this->rhs[4][pp.first][i]
                                    -  475*this->rhs[5][pp.first][i])/1440;
            break;
    }
    this->redistribute(this->state, this->rhs, this->domain_particles);
    this->roll_rhs();
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::step()
{
    TIMEZONE("rFFTW_distributed_particles::step");
    this->AdamsBashforth((this->iteration < this->integration_steps) ?
                          this->iteration+1 :
                          this->integration_steps);
    this->iteration++;
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::sort_into_domains(
        const std::map<int, single_particle_state<particle_type>> &x,
        std::map<int, std::set<int>> &dp)
{
    TIMEZONE("rFFTW_distributed_particles::sort_into_domains");
    int tmpint1, tmpint2;
    dp.clear();
    dp[-1] = std::set<int>();
    dp[ 0] = std::set<int>();
    dp[ 1] = std::set<int>();
    //dp[-1].reserve(unsigned(
    //            1.5*(interp_neighbours*2+2)*
    //            float(this->nparticles) /
    //            this->nprocs));
    //dp[ 1].reserve(unsigned(
    //            1.5*(interp_neighbours*2+2)*
    //            float(this->nparticles) /
    //            this->nprocs));
    //dp[ 0].reserve(unsigned(
    //            1.5*(this->vel->descriptor->subsizes[0] - interp_neighbours*2-2)*
    //            float(this->nparticles) /
    //            this->nprocs));
    for (auto &xx: x)
    {
        if (this->vel->get_rank_info(xx.second.data[2], tmpint1, tmpint2))
        {
            if (tmpint1 == tmpint2)
                dp[0].insert(xx.first);
            else
            {
                if (this->myrank == tmpint1)
                    dp[-1].insert(xx.first);
                else
                    dp[ 1].insert(xx.first);
            }
        }
    }
}


template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::read()
{
    TIMEZONE("rFFTW_distributed_particles::read");
    double *temp = new double[this->chunk_size*state_dimension(particle_type)];
    int tmpint1, tmpint2;
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //read state
        if (this->myrank == 0){
            TIMEZONE("read_state_chunk");
            this->read_state_chunk(cindex, temp);
        }
        {
            TIMEZONE("MPI_Bcast");
            MPI_Bcast(
                temp,
                this->chunk_size*state_dimension(particle_type),
                MPI_DOUBLE,
                0,
                this->comm);
        }
        for (unsigned int p=0; p<this->chunk_size; p++)
        {
            if (this->vel->get_rank_info(temp[state_dimension(particle_type)*p+2], tmpint1, tmpint2))
            {
                this->state[p+cindex*this->chunk_size] = temp + state_dimension(particle_type)*p;
            }
        }
        //read rhs
        if (this->iteration > 0){
            TIMEZONE("this->iteration > 0");
            for (int i=0; i<this->integration_steps; i++)
            {
                if (this->myrank == 0){
                    TIMEZONE("read_rhs_chunk");
                    this->read_rhs_chunk(cindex, i, temp);
                }
                {
                    TIMEZONE("MPI_Bcast");
                    MPI_Bcast(
                        temp,
                        this->chunk_size*state_dimension(particle_type),
                        MPI_DOUBLE,
                        0,
                        this->comm);
                }
                for (unsigned int p=0; p<this->chunk_size; p++)
                {
                    auto pp = this->state.find(p+cindex*this->chunk_size);
                    if (pp != this->state.end())
                        this->rhs[i][p+cindex*this->chunk_size] = temp + state_dimension(particle_type)*p;
                }
            }
        }
    }
    this->sort_into_domains(this->state, this->domain_particles);
    DEBUG_MSG("%s->state.size = %ld\n", this->name.c_str(), this->state.size());
    for (int domain=-1; domain<=1; domain++)
    {
        DEBUG_MSG("domain %d nparticles = %ld\n", domain, this->domain_particles[domain].size());
    }
    delete[] temp;
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const char *dset_name,
        std::map<int, single_particle_state<POINT3D>> &y)
{
    TIMEZONE("rFFTW_distributed_particles::write");
    double *data = new double[this->nparticles*3];
    double *yy = new double[this->nparticles*3];
    int pindex = 0;
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        std::fill_n(yy, this->chunk_size*3, 0);
        for (unsigned int p=0; p<this->chunk_size; p++, pindex++)
        {
            if (this->domain_particles[-1].find(pindex) != this->domain_particles[-1].end() ||
                this->domain_particles[ 0].find(pindex) != this->domain_particles[ 0].end())
            {
                std::copy(y[pindex].data,
                          y[pindex].data + 3,
                          yy + p*3);
            }
        }
        {
            TIMEZONE("MPI_Allreduce");
            MPI_Allreduce(
                yy,
                data,
                3*this->chunk_size,
                MPI_DOUBLE,
                MPI_SUM,
                this->comm);
        }
        if (this->myrank == 0){
            TIMEZONE("write_point3D_chunk");
            this->write_point3D_chunk(dset_name, cindex, data);
        }
    }
    delete[] yy;
    delete[] data;
}

template <particle_types particle_type, class rnumber, int interp_neighbours>
void rFFTW_distributed_particles<particle_type, rnumber, interp_neighbours>::write(
        const bool write_rhs)
{
    TIMEZONE("rFFTW_distributed_particles::write2");
    double *temp0 = new double[this->chunk_size*state_dimension(particle_type)];
    double *temp1 = new double[this->chunk_size*state_dimension(particle_type)];
    int pindex = 0;
    for (unsigned int cindex=0; cindex<this->get_number_of_chunks(); cindex++)
    {
        //write state
        std::fill_n(temp0, state_dimension(particle_type)*this->chunk_size, 0);
        pindex = cindex*this->chunk_size;
        for (unsigned int p=0; p<this->chunk_size; p++, pindex++)
        {
            if (this->domain_particles[-1].find(pindex) != this->domain_particles[-1].end() ||
                this->domain_particles[ 0].find(pindex) != this->domain_particles[ 0].end())
            {
                TIMEZONE("std::copy");
                std::copy(this->state[pindex].data,
                          this->state[pindex].data + state_dimension(particle_type),
                          temp0 + p*state_dimension(particle_type));
            }
        }
        {
            TIMEZONE("MPI_Allreduce");
            MPI_Allreduce(
                    temp0,
                    temp1,
                    state_dimension(particle_type)*this->chunk_size,
                    MPI_DOUBLE,
                    MPI_SUM,
                    this->comm);
        }
        if (this->myrank == 0){
            TIMEZONE("write_state_chunk");
            this->write_state_chunk(cindex, temp1);
        }
        //write rhs
        if (write_rhs){
            TIMEZONE("write_rhs");
            for (int i=0; i<this->integration_steps; i++)
            {
                std::fill_n(temp0, state_dimension(particle_type)*this->chunk_size, 0);
                pindex = cindex*this->chunk_size;
                for (unsigned int p=0; p<this->chunk_size; p++, pindex++)
                {
                    if (this->domain_particles[-1].find(pindex) != this->domain_particles[-1].end() ||
                        this->domain_particles[ 0].find(pindex) != this->domain_particles[ 0].end())
                    {
                        TIMEZONE("std::copy");
                        std::copy(this->rhs[i][pindex].data,
                                  this->rhs[i][pindex].data + state_dimension(particle_type),
                                  temp0 + p*state_dimension(particle_type));
                    }
                }
                {
                    TIMEZONE("MPI_Allreduce");
                    MPI_Allreduce(
                        temp0,
                        temp1,
                        state_dimension(particle_type)*this->chunk_size,
                        MPI_DOUBLE,
                        MPI_SUM,
                        this->comm);
                }
                if (this->myrank == 0){
                    TIMEZONE("write_rhs_chunk");
                    this->write_rhs_chunk(cindex, i, temp1);
                }
            }
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

