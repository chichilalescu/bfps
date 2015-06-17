/***********************************************************************
*
*  Copyright 2015 Johns Hopkins University
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
* Contact: turbulence@pha.jhu.edu
* Website: http://turbulence.pha.jhu.edu/
*
************************************************************************/

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "fftwf_tools.hpp"



// should I use a template here?
int fftwf_copy_complex_array(
        field_descriptor<float> *fi,
        fftwf_complex *ai,
        field_descriptor<float> *fo,
        fftwf_complex *ao)
{
    if (((fi->ndims != 3) ||
         (fo->ndims != 3)) ||
        (fi->comm != fo->comm))
        return EXIT_FAILURE;
    fftwf_complex *buffer;
    buffer = fftwf_alloc_complex(fi->slice_size);

    int min_fast_dim;
    min_fast_dim =
        (fi->sizes[fi->ndims - 1] > fo->sizes[fi->ndims - 1]) ?
         fo->sizes[fi->ndims - 1] : fi->sizes[fi->ndims - 1];

    // clean up destination, in case we're padding with zeros
    // (even if only for one dimension)
    std::fill_n((float*)ao, fo->local_size, 0.0);

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
                    (float*)(ai + (ii0 - fi->starts[0]    )*fi->slice_size),
                    (float*)(ai + (ii0 - fi->starts[0] + 1)*fi->slice_size),
                    (float*)buffer);
        }
        else
        {
            if (fi->myrank == irank)
            {
                MPI_Send(
                        (void*)(ai + (ii0-fi->starts[0])*fi->slice_size),
                        fi->slice_size,
                        MPI_COMPLEX8,
                        orank,
                        ii0,
                        fi->comm);
            }
            if (fi->myrank == orank)
            {
                MPI_Recv(
                        (void*)(buffer),
                        fi->slice_size,
                        MPI_COMPLEX8,
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
                        (float*)(buffer + ii1*fi->sizes[2]),
                        (float*)(buffer + ii1*fi->sizes[2] + min_fast_dim),
                        (float*)(ao +
                                 ((oi0 - fo->starts[0])*fo->sizes[1] +
                                  oi1)*fo->sizes[2]));
            }
        }
    }
    fftw_free(buffer);
    MPI_Barrier(fi->comm);

    return EXIT_SUCCESS;
}

int fftwf_clip_zero_padding(
        field_descriptor<float> *f,
        float *a,
        int howmany)
{
    if (f->ndims != 3)
        return EXIT_FAILURE;
    float *b = a;
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

int fftwf_get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor<float> **fr,
        field_descriptor<float> **fc)
{
    int ntmp[3];
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2;
   *fr = new field_descriptor<float>(3, ntmp, MPI_FLOAT, MPI_COMM_WORLD);
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2/2+1;
    *fc = new field_descriptor<float>(3, ntmp, MPI_COMPLEX8, MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

