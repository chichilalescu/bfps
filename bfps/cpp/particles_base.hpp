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



#include <vector>
#include <hdf5.h>
#include <unordered_map>
#include "interpolator_base.hpp"

#ifndef PARTICLES_BASE

#define PARTICLES_BASE

/* particle types */
enum particle_types {POINT3D, VELOCITY_TRACER, POINT3Dx3D};

/* space dimension */
constexpr unsigned int state_dimension(particle_types particle_type)
{
    return ((particle_type == POINT3D) ? 3 : (
            (particle_type == VELOCITY_TRACER) ? 3 : (
            (particle_type == POINT3Dx3D) ? 9 :
            -1)));
}

/* 1 particle state type */

template <particle_types particle_type>
class single_particle_state
{
    public:
        double data[state_dimension(particle_type)];

        single_particle_state();
        single_particle_state(const single_particle_state &src);
        single_particle_state(const double *src);
        ~single_particle_state();

        single_particle_state<particle_type> &operator=(const single_particle_state &src);
        single_particle_state<particle_type> &operator=(const double *src);

        inline double &operator[](const int i)
        {
            return this->data[i];
        }
};

std::vector<std::vector<hsize_t>> get_chunk_offsets(
        std::vector<hsize_t> data_dims,
        std::vector<hsize_t> chnk_dims);

template <particle_types particle_type>
class particles_io_base
{
    protected:
        int myrank, nprocs;
        MPI_Comm comm;

        unsigned int nparticles;

        std::string name;
        unsigned int chunk_size;
        int traj_skip;

        hid_t hdf5_group_id;
        std::vector<hsize_t> hdf5_state_dims, hdf5_state_chunks;
        std::vector<hsize_t> hdf5_rhs_dims, hdf5_rhs_chunks;

        std::vector<std::vector<hsize_t>> chunk_offsets;

        particles_io_base(
                const char *NAME,
                const int TRAJ_SKIP,
                const hid_t data_file_id,
                MPI_Comm COMM);
        virtual ~particles_io_base();

        void read_state_chunk(
                const int cindex,
                double *__restrict__ data);
        void write_state_chunk(
                const int cindex,
                const double *data);
        void read_rhs_chunk(
                const int cindex,
                const int rhsindex,
                double *__restrict__ data);
        void write_rhs_chunk(
                const int cindex,
                const int rhsindex,
                const double *data);

        void write_point3D_chunk(
                const std::string dset_name,
                const int cindex,
                const double *data);

    public:
        int iteration;

        inline const char *get_name()
        {
            return this->name.c_str();
        }
        inline const unsigned int get_number_of_chunks()
        {
            return this->chunk_offsets.size();
        }
        inline const unsigned int get_number_of_rhs_chunks();
        virtual void read() = 0;
        virtual void write(const bool write_rhs = true) = 0;
};

#endif//PARTICLES_BASE

