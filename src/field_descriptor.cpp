/***********************************************************************
*
*  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: Cristian.Lalescu@ds.mpg.de
*
************************************************************************/

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "base.hpp"
#include "field_descriptor.hpp"

template <class rnumber>
field_descriptor<rnumber>::field_descriptor(
        int ndims,
        int *n,
        MPI_Datatype element_type,
        MPI_Comm COMM_TO_USE)
{
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
    ptrdiff_t *nfftw = new ptrdiff_t[ndims];
    ptrdiff_t local_n0, local_0_start;
    for (int i = 0; i < this->ndims; i++)
        nfftw[i] = n[i];
    this->local_size = fftw_mpi_local_size_many(
            this->ndims,
            nfftw,
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
    tsizes[ndims-1] *= sizeof(float);
    tsubsizes[ndims-1] *= sizeof(float);
    tstarts[ndims-1] *= sizeof(float);
    if (this->mpi_dtype == MPI_COMPLEX8)
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
            "MPI_Type_create_subarray\n"
            "%d %d %d\n",
            this->sizes[0],
            this->subsizes[0],
            this->starts[0]);
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
}

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
}

template<>
int field_descriptor<float>::read(
        const char *fname,
        void *buffer)
{
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        int read_size = this->local_size*sizeof(float);
        char ffname[200];
        if (this->mpi_dtype == MPI_COMPLEX8)
            read_size *= 2;
        sprintf(ffname, "%s", fname);

        MPI_File_open(
                this->io_comm,
                ffname,
                MPI_MODE_RDONLY,
                info,
                &f);
        MPI_File_set_view(
                f,
                0,
                MPI_UNSIGNED_CHAR,
                this->mpi_array_dtype,
                "native",
                info);
        MPI_File_read_all(
                f,
                buffer,
                read_size,
                MPI_UNSIGNED_CHAR,
                MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    }
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::write(
        const char *fname,
        void *buffer)
{
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        int read_size = this->local_size*sizeof(float);
        char ffname[200];
        if (this->mpi_dtype == MPI_COMPLEX8)
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
                "native",
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

template<>
int field_descriptor<float>::transpose(
        float *input,
        float *output)
{
    // IMPORTANT NOTE:
    // for 3D transposition, the input data is messed up
    fftwf_plan tplan;
    if (this->ndims == 3)
    {
        // transpose the two local dimensions 1 and 2
        float *atmp;
        atmp = fftwf_alloc_real(this->slice_size);
        for (int k = 0; k < this->subsizes[0]; k++)
        {
            // put transposed slice in atmp
            for (int j = 0; j < this->sizes[1]; j++)
                for (int i = 0; i < this->sizes[2]; i++)
                    atmp[i*this->sizes[1] + j] =
                        input[(k*this->sizes[1] + j)*this->sizes[2] + i];
            // copy back transposed slice
            std::copy(
                    atmp,
                    atmp + this->slice_size,
                    input + k*this->slice_size);
        }
        fftwf_free(atmp);
    }
    tplan = fftwf_mpi_plan_transpose(
            this->sizes[0], this->slice_size,
            input, output,
            this->comm,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    fftwf_destroy_plan(tplan);
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::transpose(
        fftwf_complex *input,
        fftwf_complex *output)
{
    switch (this->ndims)
    {
        case 2:
            // do a global transpose over the 2 dimensions
            if (output == NULL)
            {
                std::cerr << "bad arguments for transpose.\n" << std::endl;
                return EXIT_FAILURE;
            }
            fftwf_plan tplan;
            tplan = fftwf_mpi_plan_many_transpose(
                    this->sizes[0], this->sizes[1], 2,
                    FFTW_MPI_DEFAULT_BLOCK,
                    FFTW_MPI_DEFAULT_BLOCK,
                    (float*)input, (float*)output,
                    this->comm,
                    FFTW_ESTIMATE);
            fftwf_execute(tplan);
            fftwf_destroy_plan(tplan);
            break;
        case 3:
            // transpose the two local dimensions 1 and 2
            fftwf_complex *atmp;
            atmp = fftwf_alloc_complex(this->slice_size);
            for (int k = 0; k < this->subsizes[0]; k++)
            {
                // put transposed slice in atmp
                for (int j = 0; j < this->sizes[1]; j++)
                    for (int i = 0; i < this->sizes[2]; i++)
                    {
                        atmp[i*this->sizes[1] + j][0] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][0];
                        atmp[i*this->sizes[1] + j][1] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][1];
                    }
                // copy back transposed slice
                std::copy(
                        (float*)(atmp),
                        (float*)(atmp + this->slice_size),
                        (float*)(input + k*this->slice_size));
            }
            fftwf_free(atmp);
            break;
        default:
            return EXIT_FAILURE;
            break;
    }
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::interleave(
        float *a,
        int dim)
{
/* the following is copied from
 * http://agentzlerich.blogspot.com/2010/01/using-fftw-for-in-place-matrix.html
 * */
    fftwf_iodim howmany_dims[2];
    howmany_dims[0].n  = dim;
    howmany_dims[0].is = this->local_size;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = this->local_size;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = dim;
    const int howmany_rank = sizeof(howmany_dims)/sizeof(howmany_dims[0]);

    fftwf_plan tmp = fftwf_plan_guru_r2r(
            /*rank*/0,
            /*dims*/NULL,
            howmany_rank,
            howmany_dims,
            a,
            a,
            /*kind*/NULL,
            FFTW_ESTIMATE);
    fftwf_execute(tmp);
    fftwf_destroy_plan(tmp);
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::interleave(
        fftwf_complex *a,
        int dim)
{
    fftwf_iodim howmany_dims[2];
    howmany_dims[0].n  = dim;
    howmany_dims[0].is = this->local_size;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = this->local_size;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = dim;
    const int howmany_rank = sizeof(howmany_dims)/sizeof(howmany_dims[0]);

    fftwf_plan tmp = fftwf_plan_guru_dft(
            /*rank*/0,
            /*dims*/NULL,
            howmany_rank,
            howmany_dims,
            a,
            a,
            +1,
            FFTW_ESTIMATE);
    fftwf_execute(tmp);
    fftwf_destroy_plan(tmp);
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::switch_endianness(
        float *a)
{
    for (int i = 0; i < this->local_size; i++)
    {
        *a = btle(*a);
        a++;
    }
    return EXIT_SUCCESS;
}

template<>
int field_descriptor<float>::switch_endianness(
        fftwf_complex *b)
{
    float *a = (float*)b;
    for (int i = 0; i < this->local_size; i++)
    {
        *a = btle(*a);
        a++;
        *a = btle(*a);
        a++;
    }
    return EXIT_SUCCESS;
}

template<>
field_descriptor<float>* field_descriptor<float>::get_transpose()
{
    int n[this->ndims];
    for (int i=0; i<this->ndims; i++)
        n[i] = this->sizes[this->ndims - i - 1];
    return new field_descriptor<float>(this->ndims, n, this->mpi_dtype, this->comm);
}

template class field_descriptor<float>;

