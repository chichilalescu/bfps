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
#include <vector>
#include <hdf5.h>
#include "base.hpp"
#include "particles_base.hpp"
#include "fluid_solver_base.hpp"
#include "interpolator.hpp"

#ifndef DISTRIBUTED_PARTICLES

#define DISTRIBUTED_PARTICLES

template <int particle_type, class rnumber, int interp_neighbours>
class distributed_particles
{
    public:
        int myrank, nprocs;
        MPI_Comm comm;

        std::unordered_map<int, single_particle_state<particle_type> > state;
        std::vector<std::unordered_map<int, single_particle_state<particle_type>>> rhs;
        int nparticles;
        int ncomponents;
        int integration_steps;
        int traj_skip;
        // this class only works with buffered interpolator
        interpolator<rnumber, interp_neighbours> *vel;

        /* simulation parameters */
        char name[256];
        int iteration;
        double dt;

        /* methods */

        /* constructor and destructor.
         * allocate and deallocate:
         *  this->state
         *  this->rhs
         * */
        distributed_particles(
                const char *NAME,
                interpolator<rnumber, interp_neighbours> *FIELD,
                const int NPARTICLES,
                const int TRAJ_SKIP,
                const int INTEGRATION_STEPS = 2);
        ~distributed_particles();

        void sample(
                interpolator<rnumber, interp_neighbours> *field,
                const hid_t data_file_id,
                const char *dset_name);
        void sample(
                interpolator<rnumber, interp_neighbours> *field,
                std::unordered_map<int, single_particle_state<POINT3D>> &y);
        void get_rhs(
                const std::unordered_map<int, single_particle_state<particle_type>> &x,
                std::unordered_map<int, single_particle_state<particle_type>> &y);

        void redistribute(
                std::unordered_map<int, single_particle_state<particle_type>> &x,
                std::vector<std::unordered_map<int, single_particle_state<particle_type>>> &vals);


        /* input/output */
        void read(const hid_t data_file_id);
        void write(
                const hid_t data_file_id,
                const char *dset_name,
                std::unordered_map<int, single_particle_state<POINT3D>> &y);
        void write(
                const hid_t data_file_id,
                const char *dset_name,
                std::unordered_map<int, single_particle_state<particle_type>> &y);
        void write(const hid_t data_file_id, const bool write_rhs = true);

        /* solvers */
        void step();
        void roll_rhs();
        void AdamsBashforth(const int nsteps);
};

#endif//DISTRIBUTED_PARTICLES

