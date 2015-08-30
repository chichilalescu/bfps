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
#include "fftw_tools.hpp"



template <class rnumber>
int copy_complex_array(
        field_descriptor<rnumber> *fi,
        rnumber (*ai)[2],
        field_descriptor<rnumber> *fo,
        rnumber (*ao)[2],
        int howmany)
{
    fftwf_complex *buffer;
    buffer = fftwf_alloc_complex(fi->slice_size*howmany);

    int min_fast_dim;
    min_fast_dim =
        (fi->sizes[2] > fo->sizes[2]) ?
         fo->sizes[2] : fi->sizes[2];
    MPI_Datatype MPI_CNUM = (sizeof(rnumber) == 4) ? MPI_COMPLEX : MPI_COMPLEX16;

    // clean up destination, in case we're padding with zeros
    // (even if only for one dimension)
    std::fill_n((rnumber*)ao, fo->local_size, 0.0);

    int64_t ii0, ii1;
    int64_t oi0, oi1;
    int64_t delta1, delta0;
    int irank, orank;
    delta0 = (fo->sizes[0] - fi->sizes[0]);
    delta1 = (fo->sizes[1] - fi->sizes[1]);
    for (ii0=0; ii0 < fi->sizes[0]; ii0++)
    {
        if (ii0 <= fi->sizes[0]/2)
        {
            oi0 = ii0;
            if (oi0 > fo->sizes[0]/2)
                continue;
        }
        else
        {
            oi0 = ii0 + delta0;
            if ((oi0 < 0) || ((fo->sizes[0] - oi0) >= fo->sizes[0]/2))
                continue;
        }
        irank = fi->rank[ii0];
        orank = fo->rank[oi0];
        if ((irank == orank) &&
            (irank == fi->myrank))
        {
            std::copy(
                    (rnumber*)(ai + (ii0 - fi->starts[0]    )*fi->slice_size),
                    (rnumber*)(ai + (ii0 - fi->starts[0] + 1)*fi->slice_size),
                    (rnumber*)buffer);
        }
        else
        {
            if (fi->myrank == irank)
            {
                MPI_Send(
                        (void*)(ai + (ii0-fi->starts[0])*fi->slice_size),
                        fi->slice_size,
                        MPI_CNUM,
                        orank,
                        ii0,
                        fi->comm);
            }
            if (fi->myrank == orank)
            {
                MPI_Recv(
                        (void*)(buffer),
                        fi->slice_size,
                        MPI_CNUM,
                        irank,
                        ii0,
                        fi->comm,
                        MPI_STATUS_IGNORE);
            }
        }
        if (fi->myrank == orank)
        {
            for (ii1 = 0; ii1 < fi->sizes[1]; ii1++)
            {
                if (ii1 <= fi->sizes[1]/2)
                {
                    oi1 = ii1;
                    if (oi1 > fo->sizes[1]/2)
                        continue;
                }
                else
                {
                    oi1 = ii1 + delta1;
                    if ((oi1 < 0) || ((fo->sizes[1] - oi1) >= fo->sizes[1]/2))
                        continue;
                }
                std::copy(
                        (rnumber*)(buffer + (ii1*fi->sizes[2]*howmany)),
                        (rnumber*)(buffer + (ii1*fi->sizes[2] + min_fast_dim)*howmany),
                        (rnumber*)(ao +
                                 ((oi0 - fo->starts[0])*fo->sizes[1] +
                                  oi1)*fo->sizes[2]*howmany));
            }
        }
    }
    fftw_free(buffer);
    MPI_Barrier(fi->comm);

    return EXIT_SUCCESS;
}



template <class rnumber>
int clip_zero_padding(
        field_descriptor<rnumber> *f,
        rnumber *a,
        int howmany)
{
    if (f->ndims < 3)
        return EXIT_FAILURE;
    rnumber *b = a;
    ptrdiff_t copy_size = f->sizes[2] * howmany;
    ptrdiff_t skip_size = copy_size + 2*howmany;
    for (int i0 = 0; i0 < f->subsizes[0]; i0++)
        for (int i1 = 0; i1 < f->sizes[1]; i1++)
        {
            std::copy(a, a + copy_size, b);
            a += skip_size;
            b += copy_size;
        }
    return EXIT_SUCCESS;
}



template <class rnumber>
int get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor<rnumber> **fr,
        field_descriptor<rnumber> **fc)
{
    MPI_Datatype MPI_RNUM = (sizeof(rnumber) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype MPI_CNUM = (sizeof(rnumber) == 4) ? MPI_COMPLEX : MPI_COMPLEX16;
    int ntmp[3];
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2;
    *fr = new field_descriptor<rnumber>(3, ntmp, MPI_RNUM, MPI_COMM_WORLD);
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2/2+1;
    *fc = new field_descriptor<rnumber>(3, ntmp, MPI_CNUM, MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

#define FORCE_IMPLEMENTATION(rnumber) \
template \
int copy_complex_array<rnumber>( \
        field_descriptor<rnumber> *fi, \
        rnumber (*ai)[2], \
        field_descriptor<rnumber> *fo, \
        rnumber (*ao)[2], \
        int howmany); \
 \
template \
int clip_zero_padding<rnumber>( \
        field_descriptor<rnumber> *f, \
        rnumber *a, \
        int howmany); \
 \
template \
int get_descriptors_3D<rnumber>( \
        int n0, int n1, int n2, \
        field_descriptor<rnumber> **fr, \
        field_descriptor<rnumber> **fc);

FORCE_IMPLEMENTATION(float)
FORCE_IMPLEMENTATION(double)

