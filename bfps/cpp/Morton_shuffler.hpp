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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftw_tools.hpp"

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

