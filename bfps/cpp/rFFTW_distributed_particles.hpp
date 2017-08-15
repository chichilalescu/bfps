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



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <hdf5.h>
#include "base.hpp"
#include "particles_base.hpp"
#include "fluid_solver_base.hpp"
#include "rFFTW_interpolator.hpp"

#ifndef RFFTW_DISTRIBUTED_PARTICLES

#define RFFTW_DISTRIBUTED_PARTICLES

template <particle_types particle_type, class rnumber, int interp_neighbours>
class rFFTW_distributed_particles: public particles_io_base<particle_type>
{
    private:
        // a "domain" corresponds to a region in 3D real space where a fixed set
        // of MPI processes are required to participate in the interpolation
        // formula (i.e. they all contain required information).
        // we need to know how many processes there are for each of the domains
        // to which the local process belongs.
        std::unordered_map<int, int> domain_nprocs;
        // each domain has an associated communicator, and we keep a list of the
        // communicators to which the local process belongs
        std::unordered_map<int, MPI_Comm> domain_comm;
        // for each domain, we need a list of the IDs of the particles located
        // in that domain
        std::unordered_map<int, std::unordered_set<int>> domain_particles;

        // for each domain, we need the state of each particle
        std::unordered_map<int, single_particle_state<particle_type>> state;
        // for each domain, we also need the last few values of the right hand
        // side of the ODE, since we use Adams-Bashforth integration
        std::vector<std::unordered_map<int, single_particle_state<particle_type>>> rhs;

    public:
        int integration_steps;
        // this class only works with rFFTW interpolator
        rFFTW_interpolator<rnumber, interp_neighbours> *vel;

        /* simulation parameters */
        double dt;

        /* methods */

        /* constructor and destructor.
         * allocate and deallocate:
         *  this->state
         *  this->rhs
         * */
        rFFTW_distributed_particles(
                const char *NAME,
                const hid_t data_file_id,
                rFFTW_interpolator<rnumber, interp_neighbours> *FIELD,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS = 2);
        ~rFFTW_distributed_particles();

        void sample(
                rFFTW_interpolator<rnumber, interp_neighbours> *field,
                const char *dset_name);
        void sample(
                rFFTW_interpolator<rnumber, interp_neighbours> *field,
                const std::unordered_map<int, single_particle_state<particle_type>> &x,
                const std::unordered_map<int, std::unordered_set<int>> &dp,
                std::unordered_map<int, single_particle_state<POINT3D>> &y);
        void get_rhs(
                const std::unordered_map<int, single_particle_state<particle_type>> &x,
                const std::unordered_map<int, std::unordered_set<int>> &dp,
                std::unordered_map<int, single_particle_state<particle_type>> &y);


        /* given a list of particle positions,
         * figure out which go into what local domain, and construct the relevant
         * map of ID lists "dp" (for domain particles).
         * */
        void sort_into_domains(
                const std::unordered_map<int, single_particle_state<particle_type>> &x,
                std::unordered_map<int, std::unordered_set<int>> &dp);
        /* suppose the particles are currently badly distributed, and some
         * arbitrary quantities (stored in "vals") are associated to the particles,
         * and we need to properly distribute them among processes.
         * that's what this function does.
         * In practice it's only used to redistribute the rhs values (and it
         * automatically redistributes the state x being passed).
         * Some more comments are present in the .cpp file, but, in brief: the
         * particles are simply moved from one domain to another.
         * If it turns out that the new domain contains a process which does not
         * know about a particle, that information is sent from the closest process.
         * */
        void redistribute(
                std::unordered_map<int, single_particle_state<particle_type>> &x,
                std::vector<std::unordered_map<int, single_particle_state<particle_type>>> &vals,
                std::unordered_map<int, std::unordered_set<int>> &dp);


        /* input/output */
        void read();
        void write(
                const char *dset_name,
                std::unordered_map<int, single_particle_state<POINT3D>> &y);
        void write(
                const char *dset_name,
                std::unordered_map<int, single_particle_state<particle_type>> &y);
        void write(const bool write_rhs = true);

        /* solvers */
        void step();
        void roll_rhs();
        void AdamsBashforth(const int nsteps);
};

#endif//RFFTW_DISTRIBUTED_PARTICLES

