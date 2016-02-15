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



#include <algorithm>
#include <cassert>
#include "particles_base.hpp"

template <int particle_type>
single_particle_state<particle_type>::single_particle_state()
{
    switch(particle_type)
    {
        default:
            this->data = new double[3];
            std::fill_n(this->data, 3, 0);
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type>::single_particle_state(
        const single_particle_state<particle_type> &src)
{
    switch(particle_type)
    {
        default:
            this->data = new double[3];
            std::copy(src.data, src.data + 3, this->data);
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type>::single_particle_state(
        const double *src)
{
    switch(particle_type)
    {
        default:
            this->data = new double[3];
            std::copy(src, src + 3, this->data);
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type>::~single_particle_state()
{
    switch(particle_type)
    {
        default:
            delete[] this->data;
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type> &single_particle_state<particle_type>::operator=(
        const single_particle_state &src)
{
    switch(particle_type)
    {
        default:
            std::copy(src.data, src.data + 3, this->data);
            break;
    }
    return *this;
}

template <int particle_type>
single_particle_state<particle_type> &single_particle_state<particle_type>::operator=(
        const double *src)
{
    switch(particle_type)
    {
        default:
            std::copy(src, src + 3, this->data);
            break;
    }
    return *this;
}

std::vector<std::vector<hsize_t>> get_chunk_offsets(
        std::vector<hsize_t> data_dims,
        std::vector<hsize_t> chnk_dims)
{
    std::vector<std::vector<hsize_t>> co;
    std::vector<hsize_t> nchunks(data_dims);
    int total_number_of_chunks = 1;
    for (unsigned i=0; i<nchunks.size(); i++)
    {
        nchunks[i] = data_dims[i] / chnk_dims[i];
        total_number_of_chunks *= nchunks[i];
    }
    co.resize(total_number_of_chunks);
    for (int cindex=0; cindex < total_number_of_chunks; cindex++)
    {
        int cc = cindex;
        for (unsigned i=0; i<nchunks.size(); i++)
        {
            int ii = nchunks.size()-1-i;
            co[cindex][ii] = cc % nchunks[ii];
            cc = (cc - co[cindex][ii]) / nchunks[ii];
            co[cindex][ii] *= chnk_dims[ii];
        }
    }
    return co;
}

template <int particle_type>
particles_io_base<particle_type>::particles_io_base(
        const char *NAME,
        const int TRAJ_SKIP,
        const hid_t data_file_id,
        MPI_Comm COMM)
{
    switch(particle_type)
    {
        default:
            this->ncomponents = 3;
            break;
    }
    this->name = std::string(NAME);
    this->traj_skip = TRAJ_SKIP;
    this->comm = COMM;
    MPI_Comm_rank(COMM, &this->myrank);
    MPI_Comm_size(COMM, &this->nprocs);
    hid_t dset, prop_list;

    this->hdf5_group_id = H5Gopen(data_file_id, this->name.c_str(), H5P_DEFAULT);
    std::string temp_string = (this->name + "/state");
    dset = H5Dopen(this->hdf5_group_id, temp_string.c_str(), H5P_DEFAULT);
    this->hdf5_state_dims.resize(H5Sget_simple_extent_ndims(dset));
    H5Sget_simple_extent_dims(dset, &this->hdf5_state_dims.front(), NULL);
    assert(this->hdf5_state_dims[-1] == this->ncomponents);
    this->nparticles = 1;
    for (int i=1; i<this->hdf5_state_dims.size()-1; i++)
        this->nparticles *= this->hdf5_state_dims[i];
    prop_list = H5Dget_create_plist(dset);
    this->hdf5_state_chunks.resize(this->hdf5_state_dims.size());
    H5Pget_chunk(prop_list, this->hdf5_state_dims.size(), &this->hdf5_state_chunks.front());
    H5Pclose(prop_list);
    H5Dclose(dset);
    this->chunk_size = 1;
    for (auto i=1; i<this->hdf5_state_dims.size()-1; i++)
        this->chunk_size *= this->hdf5_state_chunks[i];
    temp_string = (std::string(this->name) +
                   std::string("/rhs"));
    dset = H5Dopen(this->hdf5_group_id, temp_string.c_str(), H5P_DEFAULT);
    this->hdf5_rhs_dims.resize(H5Sget_simple_extent_ndims(dset));
    H5Sget_simple_extent_dims(dset, &this->hdf5_rhs_dims.front(), NULL);
    prop_list = H5Dget_create_plist(dset);
    this->hdf5_rhs_chunks.resize(this->hdf5_rhs_dims.size());
    H5Pget_chunk(prop_list, this->hdf5_rhs_dims.size(), &this->hdf5_rhs_chunks.front());
    H5Pclose(prop_list);
    H5Dclose(dset);

    std::vector<hsize_t> tdims(this->hdf5_state_dims), tchnk(this->hdf5_state_chunks);
    tdims.erase(tdims.begin()+0);
    tchnk.erase(tchnk.begin()+0);
    this->state_chunk_offsets = get_chunk_offsets(tdims, tchnk);
    tdims.clear();
    tchnk.clear();
    tdims = this->hdf5_rhs_dims;
    tchnk = this->hdf5_rhs_chunks;
    tdims.erase(tdims.begin()+0);
    tchnk.erase(tchnk.begin()+0);
    this->rhs_chunk_offsets = get_chunk_offsets(tdims, tchnk);
}

template <int particle_type>
particles_io_base<particle_type>::~particles_io_base()
{
    H5Gclose(this->hdf5_group_id);
}

template <int particle_type>
void particles_io_base<particle_type>::read_state_chunk(
        const int cindex,
        double *data)
{
    std::string temp_string = (std::string(this->name) +
                               std::string("/state"));
    hid_t dset = H5Dopen(this->hdf5_group_id, temp_string.c_str(), H5P_DEFAULT);
    hid_t rspace = H5Dget_space(dset);
    std::vector<hsize_t> mem_dims(this->hdf5_state_chunks);
    std::vector<hsize_t> ccindex(this->hdf5_state_chunks);
    mem_dims[0] = 1;
    hid_t mspace = H5Screate_simple(
            this->hdf5_state_dims.size(),
            &this->hdf5_state_chunks.front(),
            NULL);
    hsize_t *offset = new hsize_t[this->hdf5_state_dims.size()];
    offset[0] = this->iteration / this->traj_skip;
    for (int i=1; i<this->hdf5_state_dims.size(); i++)
        offset[i] = this->state_chunk_offsets[cindex][i-1];
    H5Sselect_hyperslab(
            rspace,
            H5S_SELECT_SET,
            offset,
            NULL,
            &this->hdf5_state_chunks.front(),
            NULL);
    H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, rspace, H5P_DEFAULT, data);
    H5Sclose(mspace);
    H5Sclose(rspace);
    H5Dclose(dset);
    delete[] offset;
}

template <int particle_type>
void particles_io_base<particle_type>::read_rhs_chunk(
        const int cindex,
        double *data)
{
}

template <int particle_type>
void particles_io_base<particle_type>::write_state_chunk(
        const int cindex,
        const double *data)
{
}

template <int particle_type>
void particles_io_base<particle_type>::write_rhs_chunk(
        const int cindex,
        const double *data)
{
}

template <int particle_type>
void particles_io_base<particle_type>::write_point3D_chunk(
        const std::string dset_name,
        const int cindex,
        const double *data)
{
}

/*****************************************************************************/
template class single_particle_state<POINT3D>;
template class single_particle_state<VELOCITY_TRACER>;

template class particles_io_base<VELOCITY_TRACER>;
/*****************************************************************************/

