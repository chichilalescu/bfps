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

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "base.hpp"
#include "field_descriptor.hpp"
#include "fftw_interface.hpp"
#include "scope_timer.hpp"

/*****************************************************************************/
/* macro for specializations to numeric types compatible with FFTW           */


template <class rnumber>
field_descriptor<rnumber>::field_descriptor(
        int ndims,
        int *n,
        MPI_Datatype element_type,
        MPI_Comm COMM_TO_USE)
{
    TIMEZONE("field_descriptor");
    DEBUG_MSG("entered field_descriptor::field_descriptor\n");
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
    this->ndims = ndims;
    this->sizes    = new int[ndims];
    this->subsizes = new int[ndims];
    this->starts   = new int[ndims];
    int tsizes   [ndims];
    int tsubsizes[ndims];
    int tstarts  [ndims];
    std::vector<ptrdiff_t> nfftw;
    nfftw.resize(ndims);
    ptrdiff_t local_n0, local_0_start;
    for (int i = 0; i < this->ndims; i++)
        nfftw[i] = n[i];
    this->local_size = fftw_interface<rnumber>::mpi_local_size_many(
                this->ndims,
                &nfftw.front(),
                1,
                FFTW_MPI_DEFAULT_BLOCK,
                this->comm,
                &local_n0,
                &local_0_start);
    this->sizes[0] = n[0];
    this->subsizes[0] = (int)local_n0;
    this->starts[0] = (int)local_0_start;
    DEBUG_MSG_WAIT(
                this->comm,
                "first subsizes[0] = %d %d %d\n",
                this->subsizes[0],
            tsubsizes[0],
            (int)local_n0);
    tsizes[0] = n[0];
    tsubsizes[0] = (int)local_n0;
    tstarts[0] = (int)local_0_start;
    DEBUG_MSG_WAIT(
                this->comm,
                "second subsizes[0] = %d %d %d\n",
                this->subsizes[0],
            tsubsizes[0],
            (int)local_n0);
    this->mpi_dtype = element_type;
    this->slice_size = 1;
    this->full_size = this->sizes[0];
    for (int i = 1; i < this->ndims; i++)
    {
        this->sizes[i] = n[i];
        this->subsizes[i] = n[i];
        this->starts[i] = 0;
        this->slice_size *= this->subsizes[i];
        this->full_size *= this->sizes[i];
        tsizes[i] = this->sizes[i];
        tsubsizes[i] = this->subsizes[i];
        tstarts[i] = this->starts[i];
    }
    tsizes[ndims-1] *= sizeof(rnumber);
    tsubsizes[ndims-1] *= sizeof(rnumber);
    tstarts[ndims-1] *= sizeof(rnumber);
    if (this->mpi_dtype == mpi_real_type<rnumber>::complex())
    {
        tsizes[ndims-1] *= 2;
        tsubsizes[ndims-1] *= 2;
        tstarts[ndims-1] *= 2;
    }
    int local_zero_array[this->nprocs], zero_array[this->nprocs];
    for (int i=0; i<this->nprocs; i++)
        local_zero_array[i] = 0;
    local_zero_array[this->myrank] = (this->subsizes[0] == 0) ? 1 : 0;
    MPI_Allreduce(
                local_zero_array,
                zero_array,
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
    int no_of_excluded_ranks = 0;
    for (int i = 0; i<this->nprocs; i++)
        no_of_excluded_ranks += zero_array[i];
    DEBUG_MSG_WAIT(
                this->comm,
                "subsizes[0] = %d %d\n",
                this->subsizes[0],
            tsubsizes[0]);
    if (no_of_excluded_ranks == 0)
    {
        this->io_comm = this->comm;
        this->io_nprocs = this->nprocs;
        this->io_myrank = this->myrank;
    }
    else
    {
        int excluded_rank[no_of_excluded_ranks];
        for (int i=0, j=0; i<this->nprocs; i++)
            if (zero_array[i])
            {
                excluded_rank[j] = i;
                j++;
            }
        MPI_Group tgroup0, tgroup;
        MPI_Comm_group(this->comm, &tgroup0);
        MPI_Group_excl(tgroup0, no_of_excluded_ranks, excluded_rank, &tgroup);
        MPI_Comm_create(this->comm, tgroup, &this->io_comm);
        MPI_Group_free(&tgroup0);
        MPI_Group_free(&tgroup);
        if (this->subsizes[0] > 0)
        {
            MPI_Comm_rank(this->io_comm, &this->io_myrank);
            MPI_Comm_size(this->io_comm, &this->io_nprocs);
        }
        else
        {
            this->io_myrank = MPI_PROC_NULL;
            this->io_nprocs = -1;
        }
    }
    DEBUG_MSG_WAIT(
                this->comm,
                "inside field_descriptor constructor, about to call "
                "MPI_Type_create_subarray "
                "%d %d %d\n",
                this->sizes[0],
            this->subsizes[0],
            this->starts[0]);
    for (int i=0; i<this->ndims; i++)
        DEBUG_MSG_WAIT(
                    this->comm,
                    "tsizes "
                    "%d %d %d\n",
                    tsizes[i],
                    tsubsizes[i],
                    tstarts[i]);
    if (this->subsizes[0] > 0)
    {
        DEBUG_MSG("creating subarray\n");
        MPI_Type_create_subarray(
                    ndims,
                    tsizes,
                    tsubsizes,
                    tstarts,
                    MPI_ORDER_C,
                    MPI_UNSIGNED_CHAR,
                    &this->mpi_array_dtype);
        MPI_Type_commit(&this->mpi_array_dtype);
    }
    this->rank = new int[this->sizes[0]];
    int *local_rank = new int[this->sizes[0]];
    std::fill_n(local_rank, this->sizes[0], 0);
    for (int i = 0; i < this->sizes[0]; i++)
        if (i >= this->starts[0] && i < this->starts[0] + this->subsizes[0])
            local_rank[i] = this->myrank;
    MPI_Allreduce(
                local_rank,
                this->rank,
                this->sizes[0],
            MPI_INT,
            MPI_SUM,
            this->comm);
    delete[] local_rank;
    this->all_start0 = new int[this->nprocs];
    int *local_start0 = new int[this->nprocs];
    std::fill_n(local_start0, this->nprocs, 0);
    for (int i = 0; i < this->nprocs; i++)
        if (this->myrank == i)
            local_start0[i] = this->starts[0];
    MPI_Allreduce(
                local_start0,
                this->all_start0,
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
    delete[] local_start0;
    this->all_size0  = new int[this->nprocs];
    int *local_size0 = new int[this->nprocs];
    std::fill_n(local_size0, this->nprocs, 0);
    for (int i = 0; i < this->nprocs; i++)
        if (this->myrank == i)
            local_size0[i] = this->subsizes[0];
    MPI_Allreduce(
                local_size0,
                this->all_size0,
                this->nprocs,
                MPI_INT,
                MPI_SUM,
                this->comm);
    delete[] local_size0;
    DEBUG_MSG("exiting field_descriptor constructor\n");
}

template <class rnumber>
int field_descriptor<rnumber>::read(
        const char *fname,
        void *buffer)
{
    TIMEZONE("field_descriptor::read");
    DEBUG_MSG("entered field_descriptor::read\n");
    char representation[] = "native";
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        ptrdiff_t read_size = this->local_size*sizeof(rnumber);
        DEBUG_MSG("read size is %ld\n", read_size);
        char ffname[200];
        if (this->mpi_dtype == mpi_real_type<rnumber>::complex())
            read_size *= 2;
        DEBUG_MSG("read size is %ld\n", read_size);
        sprintf(ffname, "%s", fname);

        MPI_File_open(
                    this->io_comm,
                    ffname,
                    MPI_MODE_RDONLY,
                    info,
                    &f);
        DEBUG_MSG("opened file\n");
        MPI_File_set_view(
                    f,
                    0,
                    MPI_UNSIGNED_CHAR,
                    this->mpi_array_dtype,
                    representation,
                    info);
        DEBUG_MSG("view is set\n");
        MPI_File_read_all(
                    f,
                    buffer,
                    read_size,
                    MPI_UNSIGNED_CHAR,
                    MPI_STATUS_IGNORE);
        DEBUG_MSG("info is read\n");
        MPI_File_close(&f);
    }
    DEBUG_MSG("finished with field_descriptor::read\n");
    return EXIT_SUCCESS;
}

template <class rnumber>
int field_descriptor<rnumber>::write(
        const char *fname,
        void *buffer)
{
    TIMEZONE("field_descriptor::write");
    char representation[] = "native";
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        ptrdiff_t read_size = this->local_size*sizeof(rnumber);
        char ffname[200];
        if (this->mpi_dtype == mpi_real_type<rnumber>::complex())
            read_size *= 2;
        sprintf(ffname, "%s", fname);

        MPI_File_open(
                    this->io_comm,
                    ffname,
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    info,
                    &f);
        MPI_File_set_view(
                    f,
                    0,
                    MPI_UNSIGNED_CHAR,
                    this->mpi_array_dtype,
                    representation,
                    info);
        MPI_File_write_all(
                    f,
                    buffer,
                    read_size,
                    MPI_UNSIGNED_CHAR,
                    MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    }

    return EXIT_SUCCESS;
}

template <class rnumber>
int field_descriptor<rnumber>::transpose(
        rnumber *input,
        rnumber *output)
{
    TIMEZONE("field_descriptor::transpose");
    /* IMPORTANT NOTE:
     for 3D transposition, the input data is messed up */
    typename fftw_interface<rnumber>::plan tplan;
    if (this->ndims == 3)
    {
        /* transpose the two local dimensions 1 and 2 */
        rnumber *atmp;
        atmp = fftw_interface<rnumber>::alloc_real(this->slice_size);
        for (int k = 0; k < this->subsizes[0]; k++)
        {
            /* put transposed slice in atmp */
            for (int j = 0; j < this->sizes[1]; j++)
                for (int i = 0; i < this->sizes[2]; i++)
                    atmp[i*this->sizes[1] + j] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i];
            /* copy back transposed slice */
            std::copy(
                        atmp,
                        atmp + this->slice_size,
                        input + k*this->slice_size);
        }
        fftw_interface<rnumber>::free(atmp);
    }
    tplan = fftw_interface<rnumber>::mpi_plan_transpose(
                this->sizes[0], this->slice_size,
            input, output,
            this->comm,
            DEFAULT_FFTW_FLAG);
    fftw_interface<rnumber>::execute(tplan);
    fftw_interface<rnumber>::destroy_plan(tplan);
    return EXIT_SUCCESS;
}

template <class rnumber>
int field_descriptor<rnumber>::transpose(
        typename fftw_interface<rnumber>::complex *input,
        typename fftw_interface<rnumber>::complex *output)
{
    TIMEZONE("field_descriptor::transpose2");
    switch (this->ndims)
    {
    case 2:
        /* do a global transpose over the 2 dimensions */
        if (output == NULL)
        {
            std::cerr << "bad arguments for transpose.\n" << std::endl;
            return EXIT_FAILURE;
        }
        typename fftw_interface<rnumber>::plan tplan;
        tplan = fftw_interface<rnumber>::mpi_plan_many_transpose(
                    this->sizes[0], this->sizes[1], 2,
                FFTW_MPI_DEFAULT_BLOCK,
                FFTW_MPI_DEFAULT_BLOCK,
                (rnumber*)input, (rnumber*)output,
                this->comm,
                DEFAULT_FFTW_FLAG);
        fftw_interface<rnumber>::execute(tplan);
        fftw_interface<rnumber>::destroy_plan(tplan);
        break;
    case 3:
        /* transpose the two local dimensions 1 and 2 */
        typename fftw_interface<rnumber>::complex *atmp;
        atmp = fftw_interface<rnumber>::alloc_complex(this->slice_size);
        for (int k = 0; k < this->subsizes[0]; k++)
        {
            /* put transposed slice in atmp */
            for (int j = 0; j < this->sizes[1]; j++)
                for (int i = 0; i < this->sizes[2]; i++)
                {
                    atmp[i*this->sizes[1] + j][0] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][0];
                    atmp[i*this->sizes[1] + j][1] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][1];
                }
            /* copy back transposed slice */
            std::copy(
                        (rnumber*)(atmp),
                        (rnumber*)(atmp + this->slice_size),
                        (rnumber*)(input + k*this->slice_size));
        }
        fftw_interface<rnumber>::free(atmp);
        break;
    default:
        return EXIT_FAILURE;
        break;
    }
    return EXIT_SUCCESS;
}

template <class rnumber>
int field_descriptor<rnumber>::interleave(
        rnumber *a,
        int dim)
{
     TIMEZONE("field_descriptor::interleav");
    /* the following is copied from
 * http://agentzlerich.blogspot.com/2010/01/using-fftw-for-in-place-matrix.html
 * */
    typename fftw_interface<rnumber>::iodim howmany_dims[2];
    howmany_dims[0].n  = dim;
    howmany_dims[0].is = this->local_size;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = this->local_size;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = dim;
    const int howmany_rank = sizeof(howmany_dims)/sizeof(howmany_dims[0]);

    typename fftw_interface<rnumber>::plan tmp = fftw_interface<rnumber>::plan_guru_r2r(
                /*rank*/0,
                /*dims*/nullptr,
                howmany_rank,
                howmany_dims,
                a,
                a,
                /*kind*/nullptr,
                DEFAULT_FFTW_FLAG);
    fftw_interface<rnumber>::execute(tmp);
    fftw_interface<rnumber>::destroy_plan(tmp);
    return EXIT_SUCCESS;
}

template <class rnumber>
int field_descriptor<rnumber>::interleave(
        typename fftw_interface<rnumber>::complex *a,
        int dim)
{
     TIMEZONE("field_descriptor::interleave2");
    typename fftw_interface<rnumber>::iodim howmany_dims[2];
    howmany_dims[0].n  = dim;
    howmany_dims[0].is = this->local_size;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = this->local_size;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = dim;
    const int howmany_rank = sizeof(howmany_dims)/sizeof(howmany_dims[0]);

    typename fftw_interface<rnumber>::plan tmp = fftw_interface<rnumber>::plan_guru_dft(
                /*rank*/0,
                /*dims*/nullptr,
                howmany_rank,
                howmany_dims,
                a,
                a,
                +1,
                DEFAULT_FFTW_FLAG);
    fftw_interface<rnumber>::execute(tmp);
    fftw_interface<rnumber>::destroy_plan(tmp);
    return EXIT_SUCCESS;
}

template <class rnumber>
field_descriptor<rnumber>* field_descriptor<rnumber>::get_transpose()
{
    TIMEZONE("field_descriptor::get_transpose");
    int n[this->ndims];
    for (int i=0; i<this->ndims; i++)
        n[i] = this->sizes[this->ndims - i - 1];
    return new field_descriptor<rnumber>(this->ndims, n, this->mpi_dtype, this->comm);
}

/*****************************************************************************/
/*****************************************************************************/



/*****************************************************************************/
/* destructor looks the same for both float and double                       */
template <class rnumber>
field_descriptor<rnumber>::~field_descriptor()
{
    DEBUG_MSG_WAIT(
                MPI_COMM_WORLD,
                this->io_comm == MPI_COMM_NULL ? "null\n" : "not null\n");
    DEBUG_MSG_WAIT(
                MPI_COMM_WORLD,
                "subsizes[0] = %d \n", this->subsizes[0]);
    if (this->subsizes[0] > 0)
    {
        DEBUG_MSG_WAIT(
                    this->io_comm,
                    "deallocating mpi_array_dtype\n");
        MPI_Type_free(&this->mpi_array_dtype);
    }
    if (this->nprocs != this->io_nprocs && this->io_myrank != MPI_PROC_NULL)
    {
        DEBUG_MSG_WAIT(
                    this->io_comm,
                    "freeing io_comm\n");
        MPI_Comm_free(&this->io_comm);
    }
    delete[] this->sizes;
    delete[] this->subsizes;
    delete[] this->starts;
    delete[] this->rank;
    delete[] this->all_start0;
    delete[] this->all_size0;
}
/*****************************************************************************/



/*****************************************************************************/
/* finally, force generation of code                                         */
template class field_descriptor<float>;
template class field_descriptor<double>;
/*****************************************************************************/

