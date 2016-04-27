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


#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "field.hpp"

template <field_components fc>
field_layout<fc>::field_layout(
        hsize_t *SIZES,
        hsize_t *SUBSIZES,
        hsize_t *STARTS,
        MPI_Comm COMM_TO_USE)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);

    std::copy(SIZES, SIZES + 3, this->sizes);
    std::copy(SUBSIZES, SUBSIZES + 3, this->subsizes);
    std::copy(STARTS, STARTS + 3, this->starts);
    if (fc == THREE || fc == THREExTHREE)
    {
        this->sizes[3] = 3;
        this->subsizes[3] = 3;
        this->starts[3] = 0;
    }
    if (fc == THREExTHREE)
    {
        this->sizes[4] = 3;
        this->subsizes[4] = 3;
        this->starts[4] = 0;
    }
    this->local_size = 1;
    this->full_size = 1;
    for (int i=0; i<ndim(fc); i++)
    {
        this->local_size *= this->subsizes[i];
        this->full_size *= this->sizes[i];
    }
    /*field will at most be distributed in 2D*/
    this->rank.resize(2);
    this->all_start.resize(2);
    this->all_size.resize(2);
    for (int i=0; i<2; i++)
    {
        this->rank[i].resize(this->sizes[i]);
        std::vector<int> local_rank;
        local_rank.resize(this->sizes[i], 0);
        for (int ii=this->starts[i]; ii<this->starts[i]+this->subsizes[i]; ii++)
            local_rank[ii] = this->myrank;
        MPI_Allreduce(
                &local_rank.front(),
                &this->rank[i].front(),
                this->sizes[i],
                MPI_INT,
                MPI_SUM,
                this->comm);
        this->all_start[i].resize(this->nprocs);
        std::vector<int> local_start;
        local_start.resize(this->nprocs, 0);
        local_start[this->myrank] = this->starts[i];
        MPI_Allreduce(
                &local_start.front(),
                &this->all_start[i].front(),
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
        this->all_size[i].resize(this->nprocs);
        std::vector<int> local_subsize;
        local_subsize.resize(this->nprocs, 0);
        local_subsize[this->myrank] = this->subsizes[i];
        MPI_Allreduce(
                &local_subsize.front(),
                &this->all_size[i].front(),
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
    }
}

template <class rnumber,
          field_representation repr,
          field_backend be,
          field_components fc>
field<rnumber, repr, be, fc>::field(
                int nx, int ny, int nz,
                MPI_Comm COMM_TO_USE)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
    switch(be)
    {
        case FFTW:
            ptrdiff_t nfftw[3];
            nfftw[0] = nz;
            nfftw[1] = ny;
            nfftw[2] = nx;
            ptrdiff_t tmp_local_size;
            ptrdiff_t local_n0, local_0_start;
            ptrdiff_t local_n1, local_1_start;
            tmp_local_size = fftw_mpi_local_size_many_transposed(
                    3, nfftw, ncomp(fc),
                    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, this->comm,
                    &local_n0, &local_0_start,
                    &local_n1, &local_1_start);
            hsize_t sizes[3], subsizes[3], starts[3];
            if (repr == REAL || repr == BOTH)
            {
                sizes[0] = nz; sizes[1] = ny; sizes[2] = nx;
                subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx;
                starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
                this->rlayout = new field_layout<fc>(
                        sizes, subsizes, starts, this->comm);
                sizes[0] = nz; sizes[1] = ny; sizes[2] = nx+2;
                subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx+2;
                starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
                this->rmemlayout = new field_layout<fc>(
                        sizes, subsizes, starts, this->comm);
                this->rdata = (rnumber*)fftw_malloc(sizeof(rnumber)*this->rmemlayout->local_size);
            }
            if (repr == COMPLEX || repr == BOTH)
            {
                sizes[0] = nz; sizes[1] = ny; sizes[2] = nx/2+1;
                subsizes[0] = local_n1; subsizes[1] = ny; subsizes[2] = nx/2+1;
                starts[0] = local_1_start; starts[1] = 0; starts[2] = 0;
                this->clayout = new field_layout<fc>(
                        sizes, subsizes, starts, this->comm);
                this->cdata = (rnumber(*)[2])fftw_malloc(2*sizeof(rnumber)*this->clayout->local_size);
            }
            break;
    }
}

template <class rnumber,
          field_representation repr,
          field_backend be,
          field_components fc>
field<rnumber, repr, be, fc>::~field()
{
    if (repr == REAL || repr == BOTH)
    {
        delete this->rlayout;
        delete this->rmemlayout;
        delete[] this->rdata;
    }
    if (repr == COMPLEX || repr == BOTH)
    {
        delete this->clayout;
        delete[] this->cdata;
    }
}

template <class rnumber,
          field_representation repr,
          field_backend be,
          field_components fc>
int field<rnumber, repr, be, fc>::io(
        const char *fname,
        const char *dset_name,
        int iteration,
        bool read)
{
    hid_t file_id, dset_id, plist_id;
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, this->comm, MPI_INFO_NULL);
    file_id = H5Fopen(fname, H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);
    dset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
    hid_t fspace, mspace;
    fspace = H5Dget_space(dset_id);
    hsize_t count[ndim(fc)+1], offset[ndim(fc)+1], dims[ndim(fc)+1];
    hsize_t memoffset[ndim(fc)+1], memshape[ndim(fc)+1];
    H5Sget_simple_extent_dims(fspace, dims, NULL);
    for (int i=0; i<ndim(fc); i++)
    {
        count[i+1] = this->rlayout->subsizes[i];
        offset[i+1] = this->rlayout->starts[i];
        assert(dims[i+1] == this->rlayout->sizes[i]);
        memshape[i+1] = this->rmemlayout->subsizes[i];
        memoffset[i+1] = 0;
        DEBUG_MSG("count[%d] = %ld, "
                  "offset[%d] = %ld, "
                  "dims[%d] = %ld, "
                  "memshape[%d] = %ld\n",
                  i+1, count[i+1],
                  i+1, offset[i+1],
                  i+1, dims[i+1],
                  i+1, memshape[i+1]);
    }
    count[0] = 1;
    offset[0] = iteration;
    memshape[0] = 1;
    memoffset[0] = 0;
    mspace = H5Screate_simple(ndim(fc)+1, memshape, NULL);
    DEBUG_MSG("selecting from file space\n");
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
    DEBUG_MSG("selecting from memory space\n");
    H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);
    if (read)
        H5Dread(dset_id, H5T_NATIVE_FLOAT, mspace, fspace, H5P_DEFAULT, this->rdata);
    else
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, mspace, fspace, H5P_DEFAULT, this->rdata);
    H5Dclose(dset_id);
    H5Sclose(fspace);
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}

template class field<float, REAL, FFTW, ONE>;
template class field<float, COMPLEX, FFTW, ONE>;
template class field<float, BOTH, FFTW, ONE>;

