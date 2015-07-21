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

#include "Morton_shuffler.hpp"
#include <iostream>


Morton_shuffler::Morton_shuffler(
        int N0, int N1, int N2,
        int d,
        int nfiles)
{
    this->d = d;
    if ((nprocs % nfiles != 0) &&
        (nfiles % nprocs != 0))
    {
        std::cerr <<
            "Number of output files incompatible with number of processes.\n"
            "Aborting.\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    if ((N0/8 % nprocs != 0) &&
        (N1/8 % nprocs != 0) &&
        (N2/8 % nprocs != 0))
    {
        std::cerr <<
            "Number of cpus incompatible with z-index representation.\n"
            "Aborting.\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    int n[4];

    // various descriptions for the real data
    n[0] = N0;
    n[1] = N1;
    n[2] = N2;
    n[3] = this->d;
    this->dinput = new field_descriptor<float>(4, n, MPI_FLOAT, MPI_COMM_WORLD);
    n[0] = N0/8;
    n[1] = N1/8;
    n[2] = N2/8;
    n[3] = 8*8*8*this->d;
    this->drcubbie = new field_descriptor<float>(4, n, MPI_FLOAT, MPI_COMM_WORLD);
    n[0] = (N0/8) * (N1/8) * (N2/8);
    n[1] = 8*8*8*this->d;
    this->dzcubbie = new field_descriptor<float>(2, n, MPI_FLOAT, MPI_COMM_WORLD);

    //set up output file descriptor
    int out_rank, out_nprocs;
    out_nprocs = nprocs/nfiles;
    if (out_nprocs == 0)
    {
        out_nprocs = 1;
        this->files_per_proc = nfiles / nprocs;
    }
    else
        this->files_per_proc = 1;
    this->out_group = myrank / out_nprocs;
    out_rank = myrank % out_nprocs;
    n[0] = ((N0/8) * (N1/8) * (N2/8)) / nfiles;
    n[1] = 8*8*8*this->d;
    MPI_Comm_split(MPI_COMM_WORLD, this->out_group, out_rank, &this->out_communicator);
    this->doutput = new field_descriptor<float>(2, n, MPI_FLOAT, this->out_communicator);
}

Morton_shuffler::~Morton_shuffler()
{
    delete this->dinput;
    delete this->drcubbie;
    delete this->dzcubbie;
    delete this->doutput;

    MPI_Comm_free(&this->out_communicator);
}

int Morton_shuffler::shuffle(
        float *a,
        const char *base_fname)
{
    // TODO: can this be done in-place?
    // shuffle into z order
    ptrdiff_t z, zz;
    int rid, zid;
    int kk;
    ptrdiff_t cubbie_size = 8*8*8*this->d;
    ptrdiff_t cc;
    float *rz = fftwf_alloc_real(cubbie_size);
    float *rtmp = fftwf_alloc_real(this->dzcubbie->local_size);
    for (int k = 0; k < this->drcubbie->sizes[0]; k++)
    {
        rid = this->drcubbie->rank[k];
        kk = k - this->drcubbie->starts[0];
        for (int j = 0; j < this->drcubbie->sizes[1]; j++)
        for (int i = 0; i < this->drcubbie->sizes[2]; i++)
        {
            z = regular_to_zindex(k, j, i);
            zid = this->dzcubbie->rank[z];
            zz = z - this->dzcubbie->starts[0];
            if (myrank == rid || myrank == zid)
            {
                // first, copy data into cubbie
                if (myrank == rid)
                    for (int tk = 0; tk < 8; tk++)
                    for (int tj = 0; tj < 8; tj++)
                    {
                        cc = (((kk*8+tk)*this->dinput->sizes[1] + (j*8+tj)) *
                              this->dinput->sizes[2] + i*8)*this->d;
                        std::copy(
                                a + cc,
                                a + cc + 8*this->d,
                                rz + (tk*8 + tj)*8*this->d);
                    }
                // now copy or send/receive to zindexed array
                if (rid == zid) std::copy(
                        rz,
                        rz + cubbie_size,
                        rtmp + zz*cubbie_size);
                else
                {
                    if (myrank == rid) MPI_Send(
                            rz,
                            cubbie_size,
                            MPI_FLOAT,
                            zid,
                            z,
                            MPI_COMM_WORLD);
                    else MPI_Recv(
                            rtmp + zz*cubbie_size,
                            cubbie_size,
                            MPI_FLOAT,
                            rid,
                            z,
                            MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE);
                }
            }
        }
    }
    fftwf_free(rz);

    char temp_char[200];
    for (int fcounter = 0; fcounter < this->files_per_proc; fcounter++)
    {
        sprintf(temp_char,
                "%s_z%.7x",
                base_fname,
                (this->files_per_proc*this->out_group + fcounter)*this->doutput->sizes[0]);
        this->doutput->write(
                temp_char,
                rtmp + fcounter*this->doutput->local_size);
    }
    fftwf_free(rtmp);
    return EXIT_SUCCESS;
}

