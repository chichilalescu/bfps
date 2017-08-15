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

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "base.hpp"
#include "fftw_tools.hpp"
#include "fftw_interface.hpp"

#define NDEBUG

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

template
int clip_zero_padding<float>(
        field_descriptor<float> *f,
        float *a,
        int howmany);

template
int clip_zero_padding<double>(
        field_descriptor<double> *f,
        double *a,
        int howmany);



template <class rnumber>
int copy_complex_array(
        field_descriptor<rnumber> *fi,
        rnumber (*ai)[2],
field_descriptor<rnumber> *fo,
rnumber (*ao)[2],
int howmany)
{
    DEBUG_MSG("entered copy_complex_array\n");
    typename fftw_interface<rnumber>::complex *buffer;
    buffer = fftw_interface<rnumber>::alloc_complex(fi->slice_size*howmany);

    int min_fast_dim;
    min_fast_dim =
            (fi->sizes[2] > fo->sizes[2]) ?
                fo->sizes[2] : fi->sizes[2];

    /* clean up destination, in case we're padding with zeros
       (even if only for one dimension) */
    std::fill_n((rnumber*)ao, fo->local_size*2, 0.0);

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
                        mpi_real_type<rnumber>::complex(),
                        orank,
                        ii0,
                        fi->comm);
            }
            if (fi->myrank == orank)
            {
                MPI_Recv(
                            (void*)(buffer),
                            fi->slice_size,
                            mpi_real_type<rnumber>::complex(),
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
    fftw_interface<rnumber>::free(buffer);
    MPI_Barrier(fi->comm);

    DEBUG_MSG("exiting copy_complex_array\n");
    return EXIT_SUCCESS;
}

template
int copy_complex_array<float>(
        field_descriptor<float> *fi,
        float (*ai)[2],
        field_descriptor<float> *fo,
        float (*ao)[2],
        int howmany);

template
int copy_complex_array<double>(
        field_descriptor<double> *fi,
        double (*ai)[2],
        field_descriptor<double> *fo,
        double (*ao)[2],
        int howmany);


template <class rnumber>
int get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor<rnumber> **fr,
        field_descriptor<rnumber> **fc)
{
    int ntmp[3];
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2;
    *fr = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::real(), MPI_COMM_WORLD);
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2/2+1;
    *fc = new field_descriptor<rnumber>(3, ntmp, mpi_real_type<rnumber>::complex(), MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

template
int get_descriptors_3D<float>(
        int n0, int n1, int n2,
        field_descriptor<float> **fr,
        field_descriptor<float> **fc);

template
int get_descriptors_3D<double>(
        int n0, int n1, int n2,
        field_descriptor<double> **fr,
        field_descriptor<double> **fc);

