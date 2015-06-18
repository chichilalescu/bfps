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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftwf_tools.hpp"

#ifndef MORTON_SHUFFLER

#define MORTON_SHUFFLER

extern int myrank, nprocs;

inline ptrdiff_t part1by2(ptrdiff_t x)
{
    ptrdiff_t n = x & 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;
    return n;
}

inline ptrdiff_t unpart1by2(ptrdiff_t z)
{
        ptrdiff_t n = z & 0x09249249;
        n = (n ^ (n >>  2)) & 0x030c30c3;
        n = (n ^ (n >>  4)) & 0x0300f00f;
        n = (n ^ (n >>  8)) & 0xff0000ff;
        n = (n ^ (n >> 16)) & 0x000003ff;
        return n;
}

inline ptrdiff_t regular_to_zindex(
        ptrdiff_t x0, ptrdiff_t x1, ptrdiff_t x2)
{
    return part1by2(x0) | (part1by2(x1) << 1) | (part1by2(x2) << 2);
}

inline void zindex_to_grid3D(
        ptrdiff_t z,
        ptrdiff_t &x0, ptrdiff_t &x1, ptrdiff_t &x2)
{
    x0 = unpart1by2(z     );
    x1 = unpart1by2(z >> 1);
    x2 = unpart1by2(z >> 2);
}

class Morton_shuffler
{
    public:
        /* members */
        int d; // number of components of the field
        // descriptor for N0 x N1 x N2 x d
        field_descriptor<float> *dinput;
        // descriptor for (N0/8) x (N1/8) x (N2/8) x 8 x 8 x 8 x d
        field_descriptor<float> *drcubbie;
        // descriptor for NZ x 8 x 8 x 8 x d
        field_descriptor<float> *dzcubbie;
        // descriptor for (NZ/nfiles) x 8 x 8 x 8 x d
        field_descriptor<float> *doutput;

        // communicator to use for output
        MPI_Comm out_communicator;
        int out_group, files_per_proc;

        /* methods */
        Morton_shuffler(
                int N0, int N1, int N2,
                int d,
                int nfiles);
        ~Morton_shuffler();

        int shuffle(
                float *regular_data,
                const char *base_fname);
};

#endif//MORTON_SHUFFLER

