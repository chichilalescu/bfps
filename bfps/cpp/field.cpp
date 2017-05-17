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


#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "field.hpp"
#include "scope_timer.hpp"
#include "shared_array.hpp"



template <typename rnumber,
          field_backend be,
          field_components fc>
field<rnumber, be, fc>::field(
                const int nx,
                const int ny,
                const int nz,
                const MPI_Comm COMM_TO_USE,
                const unsigned FFTW_PLAN_RIGOR)
{
    TIMEZONE("field::field");
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);

    this->fftw_plan_rigor = FFTW_PLAN_RIGOR;
    this->real_space_representation = true;

    /* generate HDF5 data types */
    if (typeid(rnumber) == typeid(float))
        this->rnumber_H5T = H5Tcopy(H5T_NATIVE_FLOAT);
    else if (typeid(rnumber) == typeid(double))
        this->rnumber_H5T = H5Tcopy(H5T_NATIVE_DOUBLE);
    typedef struct {
        rnumber re;   /*real part*/
        rnumber im;   /*imaginary part*/
    } tmp_complex_type;
    this->cnumber_H5T = H5Tcreate(H5T_COMPOUND, sizeof(tmp_complex_type));
    H5Tinsert(this->cnumber_H5T, "r", HOFFSET(tmp_complex_type, re), this->rnumber_H5T);
    H5Tinsert(this->cnumber_H5T, "i", HOFFSET(tmp_complex_type, im), this->rnumber_H5T);

    /* switch on backend */
    switch(be)
    {
        case FFTW:
            ptrdiff_t nfftw[3];
            nfftw[0] = nz;
            nfftw[1] = ny;
            nfftw[2] = nx;
            //ptrdiff_t tmp_local_size;
            ptrdiff_t local_n0, local_0_start;
            ptrdiff_t local_n1, local_1_start;
            //tmp_local_size = fftw_mpi_local_size_many_transposed(
            fftw_mpi_local_size_many_transposed(
                    3, nfftw, ncomp(fc),
                    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, this->comm,
                    &local_n0, &local_0_start,
                    &local_n1, &local_1_start);
            hsize_t sizes[3], subsizes[3], starts[3];
            sizes[0] = nz; sizes[1] = ny; sizes[2] = nx;
            subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx;
            starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
            this->rlayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            this->npoints = this->rlayout->full_size / ncomp(fc);
            sizes[0] = nz; sizes[1] = ny; sizes[2] = nx+2;
            subsizes[0] = local_n0; subsizes[1] = ny; subsizes[2] = nx+2;
            starts[0] = local_0_start; starts[1] = 0; starts[2] = 0;
            this->rmemlayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            sizes[0] = ny; sizes[1] = nz; sizes[2] = nx/2+1;
            subsizes[0] = local_n1; subsizes[1] = nz; subsizes[2] = nx/2+1;
            starts[0] = local_1_start; starts[1] = 0; starts[2] = 0;
            this->clayout = new field_layout<fc>(
                    sizes, subsizes, starts, this->comm);
            this->data = fftw_interface<rnumber>::alloc_real(
                    this->rmemlayout->local_size);
            memset(this->data, 0, sizeof(rnumber)*this->rmemlayout->local_size);
            this->c2r_plan = fftw_interface<rnumber>::mpi_plan_many_dft_c2r(
                    3, nfftw, ncomp(fc),
                    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                    (typename fftw_interface<rnumber>::complex*)this->data,
                    this->data,
                    this->comm,
                    this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_IN);
            this->r2c_plan = fftw_interface<rnumber>::mpi_plan_many_dft_r2c(
                    3, nfftw, ncomp(fc),
                    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                    this->data,
                    (typename fftw_interface<rnumber>::complex*)this->data,
                    this->comm,
                    this->fftw_plan_rigor | FFTW_MPI_TRANSPOSED_OUT);
            break;
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc>
field<rnumber, be, fc>::~field()
{
    /* close data types */
    H5Tclose(this->rnumber_H5T);
    H5Tclose(this->cnumber_H5T);
    switch(be)
    {
        case FFTW:
            delete this->rlayout;
            delete this->rmemlayout;
            delete this->clayout;
            fftw_interface<rnumber>::free(this->data);
            fftw_interface<rnumber>::destroy_plan(this->c2r_plan);
            fftw_interface<rnumber>::destroy_plan(this->r2c_plan);
            break;
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::ift()
{
    TIMEZONE("field::ift");
    fftw_interface<rnumber>::execute(this->c2r_plan);
    this->real_space_representation = true;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::dft()
{
    TIMEZONE("field::dft");
    fftw_interface<rnumber>::execute(this->r2c_plan);
    this->real_space_representation = false;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
int field<rnumber, be, fc>::io(
        const std::string fname,
        const std::string field_name,
        const int iteration,
        const bool read)
{
    /* file dataset has same dimensions as field */
    TIMEZONE("field::io");
    hid_t file_id, dset_id, plist_id;
    dset_id = H5I_BADID;
    std::string representation = std::string(
            this->real_space_representation ?
                "real" : "complex");
    std::string dset_name = (
            "/" + field_name +
            "/" + representation +
            "/" + std::to_string(iteration));

    /* open/create file */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, this->comm, MPI_INFO_NULL);
    bool file_exists = false;
    {
        struct stat file_buffer;
        file_exists = (stat(fname.c_str(), &file_buffer) == 0);
    }
    if (read)
    {
        assert(file_exists);
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, plist_id);
    }
    else
    {
        if (file_exists)
            file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);
        else
            file_id = H5Fcreate(fname.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    }
    assert(file_id >= 0);
    H5Pclose(plist_id);

    /* check what kind of representation is being used */
    if (read)
    {
        dset_id = H5Dopen(
                file_id,
                dset_name.c_str(),
                H5P_DEFAULT);
        assert(dset_id >= 0);
        hid_t dset_type = H5Dget_type(dset_id);
        assert(dset_type >= 0);
        bool io_for_real = (
                H5Tequal(dset_type, H5T_IEEE_F32BE) ||
                H5Tequal(dset_type, H5T_IEEE_F32LE) ||
                H5Tequal(dset_type, H5T_INTEL_F32) ||
                H5Tequal(dset_type, H5T_NATIVE_FLOAT) ||
                H5Tequal(dset_type, H5T_IEEE_F64BE) ||
                H5Tequal(dset_type, H5T_IEEE_F64LE) ||
                H5Tequal(dset_type, H5T_INTEL_F64) ||
                H5Tequal(dset_type, H5T_NATIVE_DOUBLE));
        H5Tclose(dset_type);
        assert(this->real_space_representation == io_for_real);
    }

    /* generic space initialization */
    hid_t fspace, mspace;
    hsize_t count[ndim(fc)], offset[ndim(fc)], dims[ndim(fc)];
    hsize_t memoffset[ndim(fc)], memshape[ndim(fc)];

    if (this->real_space_representation)
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count[i] = this->rlayout->subsizes[i];
            offset[i] = this->rlayout->starts[i];
            dims[i] = this->rlayout->sizes[i];
            memshape[i] = this->rmemlayout->subsizes[i];
            memoffset[i] = 0;
        }
    }
    else
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count [i] = this->clayout->subsizes[i];
            offset[i] = this->clayout->starts[i];
            dims  [i] = this->clayout->sizes[i];
            memshape [i] = count[i];
            memoffset[i] = 0;
        }
    }
    mspace = H5Screate_simple(ndim(fc), memshape, NULL);
    H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);

    /* open/create data set */
    if (read)
        fspace = H5Dget_space(dset_id);
    else
    {
        if (!H5Lexists(file_id, field_name.c_str(), H5P_DEFAULT))
        {
            hid_t gid_tmp = H5Gcreate(
                    file_id, field_name.c_str(),
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(gid_tmp);
        }

        if (!H5Lexists(file_id, (field_name + "/" + representation).c_str(), H5P_DEFAULT))
        {
            hid_t gid_tmp = H5Gcreate(
                    file_id, ("/" + field_name + "/" + representation).c_str(),
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(gid_tmp);
        }
        if (H5Lexists(file_id, dset_name.c_str(), H5P_DEFAULT))
        {
            dset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);
            fspace = H5Dget_space(dset_id);
        }
        else
        {
            fspace = H5Screate_simple(
                    ndim(fc),
                    dims,
                    NULL);
            /* chunking needs to go in here */
            dset_id = H5Dcreate(
                    file_id,
                    dset_name.c_str(),
                    (this->real_space_representation ? this->rnumber_H5T : this->cnumber_H5T),
                    fspace,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);
        }
    }
    /* both dset_id and fspace should now have sane values */

    /* check file space */
    int ndims_fspace = H5Sget_simple_extent_dims(fspace, dims, NULL);
    assert(((unsigned int)(ndims_fspace)) == ndim(fc));
    if (this->real_space_representation)
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            offset[i] = this->rlayout->starts[i];
            assert(dims[i] == this->rlayout->sizes[i]);
        }
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        if (read)
        {
            std::fill_n(this->data, this->rmemlayout->local_size, 0);
            H5Dread(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
        }
        else
        {
            assert(this->real_space_representation);
            H5Dwrite(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
        }
        H5Sclose(mspace);
    }
    else
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            offset[i] = this->clayout->starts[i];
            assert(dims[i] == this->clayout->sizes[i]);
        }
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        if (read)
        {
            std::fill_n(this->data, this->clayout->local_size*2, 0);
            H5Dread(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            this->symmetrize();
        }
        else
        {
            assert(!this->real_space_representation);
            H5Dwrite(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
        }
        H5Sclose(mspace);
    }

    H5Sclose(fspace);
    /* close data set */
    H5Dclose(dset_id);
    /* close file */
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
int field<rnumber, be, fc>::io_database(
        const std::string fname,
        const std::string field_name,
        const int toffset,
        const bool read)
{
    /* file dataset is has a time dimension as well */
    TIMEZONE("field::io_database");
    hid_t file_id, dset_id, plist_id;
    dset_id = H5I_BADID;
    std::string representation = std::string(
            this->real_space_representation ?
                "real" : "complex");
    std::string dset_name = (
            "/" + field_name +
            "/" + representation);

    /* open/create file */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, this->comm, MPI_INFO_NULL);
    bool file_exists = false;
    {
        struct stat file_buffer;
        file_exists = (stat(fname.c_str(), &file_buffer) == 0);
    }
    if (read)
    {
        assert(file_exists);
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, plist_id);
    }
    else
    {
        if (file_exists)
            file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);
        else
            file_id = H5Fcreate(fname.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    }
    H5Pclose(plist_id);

    /* check what kind of representation is being used */
    if (read)
    {
        dset_id = H5Dopen(
                file_id,
                dset_name.c_str(),
                H5P_DEFAULT);
        hid_t dset_type = H5Dget_type(dset_id);
        bool io_for_real = (
                H5Tequal(dset_type, H5T_IEEE_F32BE) ||
                H5Tequal(dset_type, H5T_IEEE_F32LE) ||
                H5Tequal(dset_type, H5T_INTEL_F32) ||
                H5Tequal(dset_type, H5T_NATIVE_FLOAT) ||
                H5Tequal(dset_type, H5T_IEEE_F64BE) ||
                H5Tequal(dset_type, H5T_IEEE_F64LE) ||
                H5Tequal(dset_type, H5T_INTEL_F64) ||
                H5Tequal(dset_type, H5T_NATIVE_DOUBLE));
        H5Tclose(dset_type);
        assert(this->real_space_representation == io_for_real);
    }

    /* generic space initialization */
    hid_t fspace, mspace;
    hsize_t count[ndim(fc)+1], offset[ndim(fc)+1], dims[ndim(fc)+1];
    hsize_t memoffset[ndim(fc)+1], memshape[ndim(fc)+1];

    int dim_counter_offset = 1;
    dim_counter_offset = 1;
    count[0] = 1;
    memshape[0] = 1;
    memoffset[0] = 0;
    if (this->real_space_representation)
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count[i+dim_counter_offset] = this->rlayout->subsizes[i];
            offset[i+dim_counter_offset] = this->rlayout->starts[i];
            dims[i+dim_counter_offset] = this->rlayout->sizes[i];
            memshape[i+dim_counter_offset] = this->rmemlayout->subsizes[i];
            memoffset[i+dim_counter_offset] = 0;
        }
        mspace = H5Screate_simple(dim_counter_offset + ndim(fc), memshape, NULL);
        H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);
    }
    else
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            count[i+dim_counter_offset] = this->clayout->subsizes[i];
            offset[i+dim_counter_offset] = this->clayout->starts[i];
            dims[i+dim_counter_offset] = this->clayout->sizes[i];
            memshape[i+dim_counter_offset] = count[i+dim_counter_offset];
            memoffset[i+dim_counter_offset] = 0;
        }
        mspace = H5Screate_simple(dim_counter_offset + ndim(fc), memshape, NULL);
        H5Sselect_hyperslab(mspace, H5S_SELECT_SET, memoffset, NULL, count, NULL);
    }

    /* open/create data set */
    if (read)
        fspace = H5Dget_space(dset_id);
    else
    {
        if (!H5Lexists(file_id, field_name.c_str(), H5P_DEFAULT))
            H5Gcreate(
                    file_id, field_name.c_str(),
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (H5Lexists(file_id, dset_name.c_str(), H5P_DEFAULT))
        {
            dset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);
            fspace = H5Dget_space(dset_id);
        }
        else
        {
            fspace = H5Screate_simple(
                    ndim(fc),
                    dims,
                    NULL);
            /* chunking needs to go in here */
            dset_id = H5Dcreate(
                    file_id,
                    dset_name.c_str(),
                    (this->real_space_representation ? this->rnumber_H5T : this->cnumber_H5T),
                    fspace,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);
        }
    }
    /* both dset_id and fspace should now have sane values */

    /* check file space */
    int ndims_fspace = H5Sget_simple_extent_dims(fspace, dims, NULL);
    assert(ndims_fspace == int(ndim(fc) + 1));
    offset[0] = toffset;
    if (this->real_space_representation)
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            offset[i+dim_counter_offset] = this->rlayout->starts[i];
            assert(dims[i+dim_counter_offset] == this->rlayout->sizes[i]);
        }
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        if (read)
        {
            std::fill_n(this->data, this->rmemlayout->local_size, 0);
            H5Dread(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            this->real_space_representation = true;
        }
        else
        {
            assert(this->real_space_representation);
            H5Dwrite(dset_id, this->rnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
        }
        H5Sclose(mspace);
    }
    else
    {
        for (unsigned int i=0; i<ndim(fc); i++)
        {
            offset[i+dim_counter_offset] = this->clayout->starts[i];
            assert(dims[i+dim_counter_offset] == this->clayout->sizes[i]);
        }
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        if (read)
        {
            H5Dread(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
            this->real_space_representation = false;
            this->symmetrize();
        }
        else
        {
            assert(!this->real_space_representation);
            H5Dwrite(dset_id, this->cnumber_H5T, mspace, fspace, H5P_DEFAULT, this->data);
        }
        H5Sclose(mspace);
    }

    H5Sclose(fspace);
    /* close data set */
    H5Dclose(dset_id);
    /* close file */
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}


template <typename rnumber,
          field_backend be,
          field_components fc>
int field<rnumber, be, fc>::write_0slice(
        const hid_t group,
        const std::string field_name,
        const int iteration)
{
    TIMEZONE("field::write_0slice");
    assert(this->real_space_representation);
    assert(fc == THREE);
    if (this->myrank == 0)
    {
        hid_t dset, wspace, mspace;
        int ndims;
        hsize_t count[4], offset[4], dims[4];
        offset[0] = iteration;
        offset[1] = 0;
        offset[2] = 0;
        offset[3] = 0;
        dset = H5Dopen(
                group,
                ("0slices/" + field_name + "/real").c_str(),
                H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        // array in memory has 2 extra x points, because FFTW
        count[0] = 1;
        count[1] = this->rmemlayout->sizes[1];
        count[2] = this->rmemlayout->sizes[2];
        count[3] = 3;
        mspace = H5Screate_simple(ndims, count, NULL);
        // array in file should not have the extra 2 points
        count[1] = this->rlayout->sizes[1];
        count[2] = this->rlayout->sizes[2];
        // select right slice in file
        H5Sselect_hyperslab(
            wspace,
            H5S_SELECT_SET,
            offset,
            NULL,
            count,
            NULL);
        offset[0] = 0;
        // select proper regions of memory
        H5Sselect_hyperslab(
            mspace,
            H5S_SELECT_SET,
            offset,
            NULL,
            count,
            NULL);
        H5Dwrite(
            dset,
            this->rnumber_H5T,
            mspace,
            wspace,
            H5P_DEFAULT,
            this->data);
        H5Dclose(dset);
        H5Sclose(mspace);
        H5Sclose(wspace);
    }
    return EXIT_SUCCESS;
}


template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::compute_rspace_xincrement_stats(
                const int xcells,
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate)
{
    TIMEZONE("field::compute_rspace_xincrement_stats");
    assert(this->real_space_representation);
    assert(fc == ONE || fc == THREE);
    field<rnumber, be, fc> *tmp_field = new field<rnumber, be, fc>(
            this->rlayout->sizes[2],
            this->rlayout->sizes[1],
            this->rlayout->sizes[0],
            this->rlayout->comm);
    tmp_field->real_space_representation = true;
    this->RLOOP(
                [&](ptrdiff_t rindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
            hsize_t rrindex = (xindex + xcells)%this->rlayout->sizes[2] + (
                zindex * this->rlayout->subsizes[1] + yindex)*(
                    this->rmemlayout->subsizes[2]);
            for (unsigned int component=0; component < ncomp(fc); component++)
                tmp_field->data[rindex*ncomp(fc) + component] =
                    this->data[rrindex*ncomp(fc) + component] -
                    this->data[rindex*ncomp(fc) + component];
                    });
    tmp_field->compute_rspace_stats(
            group,
            dset_name,
            toffset,
            max_estimate);
    delete tmp_field;
}



template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::compute_rspace_stats(
                const hid_t group,
                const std::string dset_name,
                const hsize_t toffset,
                const std::vector<double> max_estimate)
{
    TIMEZONE("field::compute_rspace_stats");
    assert(this->real_space_representation);
    const unsigned int nmoments = 10;
    int nvals, nbins;
    if (this->myrank == 0)
    {
        hid_t dset, wspace;
        hsize_t dims[ndim(fc)-1];
        int ndims;
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == int(ndim(fc))-1);
        assert(dims[1] == nmoments);
        switch(ndims)
        {
            case 2:
                nvals = 1;
                break;
            case 3:
                nvals = dims[2];
                break;
            case 4:
                nvals = dims[2]*dims[3];
                break;
        }
        H5Sclose(wspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
        assert(ndims == int(ndim(fc))-1);
        nbins = dims[1];
        if (ndims == 3)
            assert(nvals == int(dims[2]));
        else if (ndims == 4)
            assert(nvals == int(dims[2]*dims[3]));
        H5Sclose(wspace);
        H5Dclose(dset);
    }
    {
        TIMEZONE("MPI_Bcast");
        MPI_Bcast(&nvals, 1, MPI_INT, 0, this->comm);
        MPI_Bcast(&nbins, 1, MPI_INT, 0, this->comm);
    }
    assert(nvals == int(max_estimate.size()));

    shared_array<double> local_moments_threaded(nmoments*nvals, [&](double* local_moments){
        std::fill_n(local_moments, nmoments*nvals, 0);
        if (nvals == 4) local_moments[3] = max_estimate[3];
    });

    shared_array<double> val_tmp_threaded(nvals,[&](double *val_tmp){
        std::fill_n(val_tmp, nvals, 0);
    });

    shared_array<ptrdiff_t> local_hist_threaded(nbins*nvals,[&](ptrdiff_t* local_hist){
        std::fill_n(local_hist, nbins*nvals, 0);
    });

    shared_array<double> local_pow_tmp(nvals);

    double *binsize = new double[nvals];
    for (int i=0; i<nvals; i++)
        binsize[i] = 2*max_estimate[i] / nbins;

    {
        TIMEZONE("field::RLOOP");
        this->RLOOP(
                [&](ptrdiff_t rindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex){
            double* pow_tmp = local_pow_tmp.getMine();
            std::fill_n(pow_tmp, nvals, 0);

            double *local_moments = local_moments_threaded.getMine();
            double *val_tmp = val_tmp_threaded.getMine();
            ptrdiff_t *local_hist = local_hist_threaded.getMine();

            if (nvals == int(4)) val_tmp[3] = 0.0;
            for (unsigned int i=0; i<ncomp(fc); i++)
            {
                val_tmp[i] = this->data[rindex*ncomp(fc)+i];
                if (nvals == int(4)) val_tmp[3] += val_tmp[i]*val_tmp[i];
            }
            if (nvals == int(4))
            {
                val_tmp[3] = sqrt(val_tmp[3]);
                if (val_tmp[3] < local_moments[0*nvals+3])
                    local_moments[0*nvals+3] = val_tmp[3];
                if (val_tmp[3] > local_moments[9*nvals+3])
                    local_moments[9*nvals+3] = val_tmp[3];
                int bin = int(floor(val_tmp[3]*2/binsize[3]));
                if (bin >= 0 && bin < nbins){
                    local_hist[bin*nvals+3]++;
                }
            }
            for (unsigned int i=0; i<ncomp(fc); i++)
            {
                if (val_tmp[i] < local_moments[0*nvals+i])
                    local_moments[0*nvals+i] = val_tmp[i];
                if (val_tmp[i] > local_moments[(nmoments-1)*nvals+i])
                    local_moments[(nmoments-1)*nvals+i] = val_tmp[i];
                int bin = int(floor((val_tmp[i] + max_estimate[i]) / binsize[i]));
                if (bin >= 0 && bin < nbins)
                    local_hist[bin*nvals+i]++;
            }
            for (int n=1; n < int(nmoments)-1; n++){
                for (int i=0; i<nvals; i++){
                    local_moments[n*nvals + i] += (pow_tmp[i] = val_tmp[i]*pow_tmp[i]);
				}
			}
                });

          TIMEZONE("FIELD_RLOOP::Merge");
          local_moments_threaded.mergeParallel([&](const int idx, const double& v1, const double& v2) -> double {
              if(nvals == int(4) && idx == 0*nvals+3){
                  return std::min(v1, v2);  
              }
              if(nvals == int(4) && idx == 9*nvals+3){
                  return std::max(v1, v2);  
              }
              if(idx < int(ncomp(fc))){
                  return std::min(v1, v2);        
              }      
              if(int(nmoments-1)*nvals <= idx && idx < int(int(nmoments-1)*nvals+ncomp(fc))){
                  return std::max(v1, v2);        
              }
              return v1 + v2;
          });

        local_hist_threaded.mergeParallel();
    }
    ptrdiff_t *hist = new ptrdiff_t[nbins*nvals];
    double *moments = new double[nmoments*nvals];
    {
        TIMEZONE("MPI_Allreduce");
        MPI_Allreduce(
                (void*)local_moments_threaded.getMasterData(),
                (void*)moments,
                nvals,
                MPI_DOUBLE, MPI_MIN, this->comm);
        MPI_Allreduce(
                (void*)(local_moments_threaded.getMasterData() + nvals),
                (void*)(moments+nvals),
                (nmoments-2)*nvals,
                MPI_DOUBLE, MPI_SUM, this->comm);
        MPI_Allreduce(
                (void*)(local_moments_threaded.getMasterData() + (nmoments-1)*nvals),
                (void*)(moments+(nmoments-1)*nvals),
                nvals,
                MPI_DOUBLE, MPI_MAX, this->comm);
        MPI_Allreduce(
                (void*)local_hist_threaded.getMasterData(),
                (void*)hist,
                nbins*nvals,
                MPI_INT64_T, MPI_SUM, this->comm);
    }
    for (int n=1; n < int(nmoments)-1; n++)
        for (int i=0; i<nvals; i++)
            moments[n*nvals + i] /= this->npoints;

    delete[] binsize;
    if (this->myrank == 0)
    {
        TIMEZONE("root-work");
        hid_t dset, wspace, mspace;
        hsize_t count[ndim(fc)-1], offset[ndim(fc)-1], dims[ndim(fc)-1];
        dset = H5Dopen(group, ("moments/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        H5Sget_simple_extent_dims(wspace, dims, NULL);
        offset[0] = toffset;
        offset[1] = 0;
        count[0] = 1;
        count[1] = nmoments;
        if (fc == THREE)
        {
            offset[2] = 0;
            count[2] = nvals;
        }
        if (fc == THREExTHREE)
        {
            offset[2] = 0;
            count[2] = 3;
            offset[3] = 0;
            count[3] = 3;
        }
        mspace = H5Screate_simple(ndim(fc)-1, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, moments);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
        dset = H5Dopen(group, ("histograms/" + dset_name).c_str(), H5P_DEFAULT);
        wspace = H5Dget_space(dset);
        count[1] = nbins;
        mspace = H5Screate_simple(ndim(fc)-1, count, NULL);
        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dwrite(dset, H5T_NATIVE_INT64, mspace, wspace, H5P_DEFAULT, hist);
        H5Sclose(wspace);
        H5Sclose(mspace);
        H5Dclose(dset);
        if (H5Lexists(group, "0slices", H5P_DEFAULT))
            this->write_0slice(
                    group,
                    dset_name,
                    toffset);
    }
    delete[] moments;
    delete[] hist;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::normalize()
{
        for (hsize_t tmp_index=0; tmp_index<this->rmemlayout->local_size; tmp_index++)
            this->data[tmp_index] /= this->npoints;
}

template <typename rnumber,
          field_backend be,
          field_components fc>
void field<rnumber, be, fc>::symmetrize()
{
    TIMEZONE("field::symmetrize");
    assert(!this->real_space_representation);
    ptrdiff_t ii, cc;
    typename fftw_interface<rnumber>::complex *data = this->get_cdata();
    MPI_Status *mpistatus = new MPI_Status;
    if (this->myrank == this->clayout->rank[0][0])
    {
        for (cc = 0; cc < ncomp(fc); cc++)
            data[cc][1] = 0.0;
        for (ii = 1; ii < ptrdiff_t(this->clayout->sizes[1]/2); ii++)
            for (cc = 0; cc < ncomp(fc); cc++) {
                ( *(data + cc + ncomp(fc)*(this->clayout->sizes[1] - ii)*this->clayout->sizes[2]))[0] =
                 (*(data + cc + ncomp(fc)*(                          ii)*this->clayout->sizes[2]))[0];
                ( *(data + cc + ncomp(fc)*(this->clayout->sizes[1] - ii)*this->clayout->sizes[2]))[1] =
                -(*(data + cc + ncomp(fc)*(                          ii)*this->clayout->sizes[2]))[1];
            }
    }
    typename fftw_interface<rnumber>::complex *buffer;
    buffer = fftw_interface<rnumber>::alloc_complex(ncomp(fc)*this->clayout->sizes[1]);
    ptrdiff_t yy;
    /*ptrdiff_t tindex;*/
    int ranksrc, rankdst;
    for (yy = 1; yy < ptrdiff_t(this->clayout->sizes[0]/2); yy++) {
        ranksrc = this->clayout->rank[0][yy];
        rankdst = this->clayout->rank[0][this->clayout->sizes[0] - yy];
        if (this->clayout->myrank == ranksrc)
            for (ii = 0; ii < ptrdiff_t(this->clayout->sizes[1]); ii++)
                for (cc = 0; cc < ncomp(fc); cc++)
                    for (int imag_comp=0; imag_comp<2; imag_comp++)
                        (*(buffer + ncomp(fc)*ii+cc))[imag_comp] =
                            (*(data + ncomp(fc)*((yy - this->clayout->starts[0])*this->clayout->sizes[1] + ii)*this->clayout->sizes[2] + cc))[imag_comp];
        if (ranksrc != rankdst)
        {
            if (this->clayout->myrank == ranksrc)
                MPI_Send((void*)buffer,
                         ncomp(fc)*this->clayout->sizes[1], mpi_real_type<rnumber>::complex(), rankdst, yy,
                        this->clayout->comm);
            if (this->clayout->myrank == rankdst)
                MPI_Recv((void*)buffer,
                         ncomp(fc)*this->clayout->sizes[1], mpi_real_type<rnumber>::complex(), ranksrc, yy,
                        this->clayout->comm, mpistatus);
        }
        if (this->clayout->myrank == rankdst)
        {
            for (ii = 1; ii < ptrdiff_t(this->clayout->sizes[1]); ii++)
                for (cc = 0; cc < ncomp(fc); cc++)
                {
                    (*(data + ncomp(fc)*((this->clayout->sizes[0] - yy - this->clayout->starts[0])*this->clayout->sizes[1] + ii)*this->clayout->sizes[2] + cc))[0] =
                            (*(buffer + ncomp(fc)*(this->clayout->sizes[1]-ii)+cc))[0];
                    (*(data + ncomp(fc)*((this->clayout->sizes[0] - yy - this->clayout->starts[0])*this->clayout->sizes[1] + ii)*this->clayout->sizes[2] + cc))[1] =
                            -(*(buffer + ncomp(fc)*(this->clayout->sizes[1]-ii)+cc))[1];
                }
            for (cc = 0; cc < ncomp(fc); cc++)
            {
                (*((data + cc + ncomp(fc)*(this->clayout->sizes[0] - yy - this->clayout->starts[0])*this->clayout->sizes[1]*this->clayout->sizes[2])))[0] =  (*(buffer + cc))[0];
                (*((data + cc + ncomp(fc)*(this->clayout->sizes[0] - yy - this->clayout->starts[0])*this->clayout->sizes[1]*this->clayout->sizes[2])))[1] = -(*(buffer + cc))[1];
            }
        }
    }
    fftw_interface<rnumber>::free(buffer);
    delete mpistatus;
    /* put asymmetric data to 0 */
    /*if (this->clayout->myrank == this->clayout->rank[0][this->clayout->sizes[0]/2])
    {
        tindex = ncomp(fc)*(this->clayout->sizes[0]/2 - this->clayout->starts[0])*this->clayout->sizes[1]*this->clayout->sizes[2];
        for (ii = 0; ii < this->clayout->sizes[1]; ii++)
        {
            std::fill_n((rnumber*)(data + tindex), ncomp(fc)*2*this->clayout->sizes[2], 0.0);
            tindex += ncomp(fc)*this->clayout->sizes[2];
        }
    }
    tindex = ncomp(fc)*();
    std::fill_n((rnumber*)(data + tindex), ncomp(fc)*2, 0.0);*/
}

template <typename rnumber,
          field_backend be,
          field_components fc>
template <kspace_dealias_type dt>
void field<rnumber, be, fc>::compute_stats(
        kspace<be, dt> *kk,
        const hid_t group,
        const std::string dset_name,
        const hsize_t toffset,
        const double max_estimate)
{
    TIMEZONE("field::compute_stats");
    std::vector<double> max_estimate_vector;
    bool did_rspace = false;
    switch(fc)
    {
        case ONE:
            max_estimate_vector.resize(1, max_estimate);
            break;
        case THREE:
            max_estimate_vector.resize(4, max_estimate);
            max_estimate_vector[3] *= sqrt(3);
            break;
        case THREExTHREE:
            max_estimate_vector.resize(9, max_estimate);
            break;
    }
    if (this->real_space_representation)
    {
        TIMEZONE("field::compute_stats::compute_rspace_stats");
        this->compute_rspace_stats(
                group,
                dset_name,
                toffset,
                max_estimate_vector);
        did_rspace = true;
        this->dft();
        // normalize
        TIMEZONE("field::normalize");
        for (hsize_t tmp_index=0; tmp_index<this->rmemlayout->local_size; tmp_index++)
            this->data[tmp_index] /= this->npoints;
    }
    // what follows gave me a headache until I found this link:
    // http://stackoverflow.com/questions/8256636/expected-primary-expression-error-on-template-method-using
    kk->template cospectrum<rnumber, fc>(
            (typename fftw_interface<rnumber>::complex*)this->data,
            (typename fftw_interface<rnumber>::complex*)this->data,
            group,
            dset_name + "_" + dset_name,
            toffset);
    if (!did_rspace)
    {
        this->ift();
        // normalization not required
        this->compute_rspace_stats(
                group,
                dset_name,
                toffset,
                max_estimate_vector);
    }
}

template <typename rnumber,
          field_backend be,
          field_components fc1,
          field_components fc2,
          kspace_dealias_type dt>
void compute_gradient(
        kspace<be, dt> *kk,
        field<rnumber, be, fc1> *src,
        field<rnumber, be, fc2> *dst)
{
    TIMEZONE("compute_gradient");
    assert(!src->real_space_representation);
    assert((fc1 == ONE && fc2 == THREE) ||
           (fc1 == THREE && fc2 == THREExTHREE));
    kk->CLOOP_K2(
            [&](ptrdiff_t cindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex,
                double k2){
            if (k2 < kk->kM2) switch(fc1)
            {
                case ONE:
                    dst->cval(cindex, 0, 0) = -kk->kx[xindex]*src->cval(cindex, 1);
                    dst->cval(cindex, 0, 1) =  kk->kx[xindex]*src->cval(cindex, 0);
                    dst->cval(cindex, 1, 0) = -kk->ky[yindex]*src->cval(cindex, 1);
                    dst->cval(cindex, 1, 1) =  kk->ky[yindex]*src->cval(cindex, 0);
                    dst->cval(cindex, 2, 0) = -kk->kz[zindex]*src->cval(cindex, 1);
                    dst->cval(cindex, 2, 1) =  kk->kz[zindex]*src->cval(cindex, 0);
                    break;
                case THREE:
                    for (unsigned int field_component = 0;
                         field_component < ncomp(fc1);
                         field_component++)
                    {
                        dst->cval(cindex, 0, field_component, 0) = -kk->kx[xindex]*src->cval(cindex, field_component, 1);
                        dst->cval(cindex, 0, field_component, 1) =  kk->kx[xindex]*src->cval(cindex, field_component, 0);
                        dst->cval(cindex, 1, field_component, 0) = -kk->ky[yindex]*src->cval(cindex, field_component, 1);
                        dst->cval(cindex, 1, field_component, 1) =  kk->ky[yindex]*src->cval(cindex, field_component, 0);
                        dst->cval(cindex, 2, field_component, 0) = -kk->kz[zindex]*src->cval(cindex, field_component, 1);
                        dst->cval(cindex, 2, field_component, 1) =  kk->kz[zindex]*src->cval(cindex, field_component, 0);
                    }
                //dst->get_cdata()[(cindex*3+0)*ncomp(fc1)+field_component][0] =
                //    - kk->kx[xindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][1];
                //dst->get_cdata()[(cindex*3+0)*ncomp(fc1)+field_component][1] =
                //      kk->kx[xindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][0];
                //dst->get_cdata()[(cindex*3+1)*ncomp(fc1)+field_component][0] =
                //    - kk->ky[yindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][1];
                //dst->get_cdata()[(cindex*3+1)*ncomp(fc1)+field_component][1] =
                //      kk->ky[yindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][0];
                //dst->get_cdata()[(cindex*3+2)*ncomp(fc1)+field_component][0] =
                //    - kk->kz[zindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][1];
                //dst->get_cdata()[(cindex*3+2)*ncomp(fc1)+field_component][1] =
                //      kk->kz[zindex]*src->get_cdata()[cindex*ncomp(fc1)+field_component][0];
            }
            });
}

template class field<float, FFTW, ONE>;
template class field<float, FFTW, THREE>;
template class field<float, FFTW, THREExTHREE>;
template class field<double, FFTW, ONE>;
template class field<double, FFTW, THREE>;
template class field<double, FFTW, THREExTHREE>;

template void field<float, FFTW, ONE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREExTHREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<double, FFTW, ONE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREExTHREE>::compute_stats<TWO_THIRDS>(
        kspace<FFTW, TWO_THIRDS> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<float, FFTW, ONE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<float, FFTW, THREExTHREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);

template void field<double, FFTW, ONE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);
template void field<double, FFTW, THREExTHREE>::compute_stats<SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        const hid_t, const std::string, const hsize_t, const double);

template void compute_gradient<float, FFTW, THREE, THREExTHREE, SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        field<float, FFTW, THREE> *,
        field<float, FFTW, THREExTHREE> *);
template void compute_gradient<double, FFTW, THREE, THREExTHREE, SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        field<double, FFTW, THREE> *,
        field<double, FFTW, THREExTHREE> *);

template void compute_gradient<float, FFTW, ONE, THREE, SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        field<float, FFTW, ONE> *,
        field<float, FFTW, THREE> *);
template void compute_gradient<double, FFTW, ONE, THREE, SMOOTH>(
        kspace<FFTW, SMOOTH> *,
        field<double, FFTW, ONE> *,
        field<double, FFTW, THREE> *);
