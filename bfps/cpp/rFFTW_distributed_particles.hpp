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

template <int particle_type, class rnumber, int interp_neighbours>
class rFFTW_distributed_particles: public particles_io_base<particle_type>
{
    private:
        std::unordered_map<int, single_particle_state<particle_type>> state;
        std::vector<std::unordered_map<int, single_particle_state<particle_type>>> rhs;
        std::unordered_map<int, int> domain_nprocs;
        std::unordered_map<int, MPI_Comm> domain_comm;
        std::unordered_map<int, std::unordered_set<int>> domain_particles;

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


        void sort_into_domains(
                const std::unordered_map<int, single_particle_state<particle_type>> &x,
                std::unordered_map<int, std::unordered_set<int>> &dp);
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

