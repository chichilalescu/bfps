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
#include <array>
#include "base.hpp"
#include "scope_timer.hpp"
#include "field_binary_IO.hpp"

template <typename rnumber, field_representation fr, field_components fc>
field_binary_IO<rnumber, fr, fc>::field_binary_IO(
                const hsize_t *SIZES,
                const hsize_t *SUBSIZES,
                const hsize_t *STARTS,
                const MPI_Comm COMM_TO_USE):
            field_layout<fc>(
                    SIZES,
                    SUBSIZES,
                    STARTS,
                    COMM_TO_USE)
{
    TIMEZONE("field_binary_IO::field_binary_IO");
    std::vector<int> tsizes   ;
    std::vector<int> tsubsizes;
    std::vector<int> tstarts  ;
    tsizes.resize(ndim(fc));
    tsubsizes.resize(ndim(fc));
    tstarts.resize(ndim(fc));
    for (int i=0; i<int(ndim(fc)); i++)
    {
        tsizes[i] = int(this->sizes[i]);
        tsubsizes[i] = int(this->subsizes[i]);
        tstarts[i] = int(this->starts[i]);
    }
    // these are required if using unsigned char in the subarray creation
    //tsizes[ndim(fc)-1] *= sizeof(element_type);
    //tsubsizes[ndim(fc)-1] *= sizeof(element_type);
    //tstarts[ndim(fc)-1] *= sizeof(element_type);
    MPI_Type_create_subarray(
                ndim(fc),
                &tsizes.front(),
                &tsubsizes.front(),
                &tstarts.front(),
                MPI_ORDER_C,
                //MPI_UNSIGNED_CHAR, // in case element type fails
                mpi_type<rnumber>(fr),
                &this->mpi_array_dtype);
    MPI_Type_commit(&this->mpi_array_dtype);

    // check if there are processes without any data
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
        this->io_comm_nprocs = this->nprocs;
        this->io_comm_myrank = this->myrank;
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
            MPI_Comm_rank(this->io_comm, &this->io_comm_myrank);
            MPI_Comm_size(this->io_comm, &this->io_comm_nprocs);
        }
        else
        {
            this->io_comm_myrank = MPI_PROC_NULL;
            this->io_comm_nprocs = -1;
        }
    }
}

template <typename rnumber, field_representation fr, field_components fc>
field_binary_IO<rnumber, fr, fc>::~field_binary_IO()
{
    TIMEZONE("field_binary_IO::~field_binary_IO");
    MPI_Type_free(&this->mpi_array_dtype);
    if (this->nprocs != this->io_comm_nprocs &&
        this->io_comm_myrank != MPI_PROC_NULL)
    {
        MPI_Comm_free(&this->io_comm);
    }
}

template <typename rnumber, field_representation fr, field_components fc>
int field_binary_IO<rnumber, fr, fc>::read(
        const std::string fname,
        void *buffer)
{
    TIMEZONE("field_binary_IO::read");
    char representation[] = "native";
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        char ffname[512];
        sprintf(ffname, "%s", fname.c_str());

        MPI_File_open(
                    this->io_comm,
                    ffname,
                    MPI_MODE_RDONLY,
                    info,
                    &f);
        MPI_File_set_view(
                    f,
                    0,
                    mpi_type<rnumber>(fr),
                    this->mpi_array_dtype,
                    representation,
                    info);
        MPI_File_read_all(
                    f,
                    buffer,
                    this->local_size,
                    mpi_type<rnumber>(fr),
                    MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    }
    return EXIT_SUCCESS;
}

template <typename rnumber, field_representation fr, field_components fc>
int field_binary_IO<rnumber, fr, fc>::write(
        const std::string fname,
        void *buffer)
{
    TIMEZONE("field_binary_IO::write");
    char representation[] = "native";
    if (this->subsizes[0] > 0)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_File f;
        char ffname[512];
        sprintf(ffname, "%s", fname.c_str());

        MPI_File_open(
                    this->io_comm,
                    ffname,
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    info,
                    &f);
        MPI_File_set_view(
                    f,
                    0,
                    mpi_type<rnumber>(fr),
                    this->mpi_array_dtype,
                    representation,
                    info);
        MPI_File_write_all(
                    f,
                    buffer,
                    this->local_size,
                    mpi_type<rnumber>(fr),
                    MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    }

    return EXIT_SUCCESS;
}

template class field_binary_IO<float , REAL   , ONE>;
template class field_binary_IO<float , COMPLEX, ONE>;
template class field_binary_IO<double, REAL   , ONE>;
template class field_binary_IO<double, COMPLEX, ONE>;

template class field_binary_IO<float , REAL   , THREE>;
template class field_binary_IO<float , COMPLEX, THREE>;
template class field_binary_IO<double, REAL   , THREE>;
template class field_binary_IO<double, COMPLEX, THREE>;

template class field_binary_IO<float , REAL   , THREExTHREE>;
template class field_binary_IO<float , COMPLEX, THREExTHREE>;
template class field_binary_IO<double, REAL   , THREExTHREE>;
template class field_binary_IO<double, COMPLEX, THREExTHREE>;

