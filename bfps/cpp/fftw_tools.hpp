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



#include <mpi.h>
#include <fftw3-mpi.h>
#include "field_descriptor.hpp"

#ifndef FFTW_TOOLS

#define FFTW_TOOLS

extern int myrank, nprocs;

/* given two arrays of the same dimension, we do a simple resize in
 * Fourier space: either chop off high modes, or pad with zeros.
 * the arrays are assumed to use 3D mpi fftw layout.
 * */
template <class rnumber>
int copy_complex_array(
        field_descriptor<rnumber> *fi,
        rnumber (*ai)[2],
        field_descriptor<rnumber> *fo,
        rnumber (*ao)[2],
        int howmany=1);

template <class rnumber>
int clip_zero_padding(
        field_descriptor<rnumber> *f,
        rnumber *a,
        int howmany=1);

/* function to get pair of descriptors for real and Fourier space
 * arrays used with fftw.
 * the n0, n1, n2 correspond to the real space data WITHOUT the zero
 * padding that FFTW needs.
 * IMPORTANT: the real space array must be allocated with
 * 2*fc->local_size, and then the zeros cleaned up before trying
 * to write data.
 * */
template <class rnumber>
int get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor<rnumber> **fr,
        field_descriptor<rnumber> **fc);

#endif//FFTW_TOOLS

