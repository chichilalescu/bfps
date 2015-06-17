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

#include <mpi.h>
#include <fftw3-mpi.h>

#ifndef FIELD_DESCRIPTOR

#define FIELD_DESCRIPTOR

extern int myrank, nprocs;

template <class rnumber>
class field_descriptor
{
    private:
        typedef rnumber cnumber[2];
    public:

        /* data */
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int *rank;
        ptrdiff_t slice_size, local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;
        int myrank, nprocs, io_myrank, io_nprocs;
        MPI_Comm comm, io_comm;


        /* methods */
        field_descriptor(
                int ndims,
                int *n,
                MPI_Datatype element_type,
                MPI_Comm COMM_TO_USE);
        ~field_descriptor();

        /* io is performed using MPI_File stuff, and our
         * own mpi_array_dtype that was defined in the constructor.
         * */
        int read(
                const char *fname,
                void *buffer);
        int write(
                const char *fname,
                void *buffer);

        /* a function that generates the transposed descriptor.
         * don't forget to delete the result once you're done with it.
         * the transposed descriptor is useful for io operations.
         * */
        field_descriptor<rnumber> *get_transpose();

        /* we don't actually need the transposed descriptor to perform
         * the transpose operation: we only need the in/out fields.
         * */
        int transpose(
                rnumber *input,
                rnumber *output);
        int transpose(
                cnumber *input,
                cnumber *output = NULL);

        int interleave(
                rnumber *input,
                int dim);
        int interleave(
                cnumber *input,
                int dim);

        int switch_endianness(
                rnumber *a);
        int switch_endianness(
                cnumber *a);
};


/* given two arrays of the same dimension, we do simple resizes in
 * Fourier space: either chop off high modes, or pad with zeros.
 * the arrays are assumed to use fftw layout.
 * */
int fftwf_copy_complex_array(
        field_descriptor<float> *fi,
        fftwf_complex *ai,
        field_descriptor<float> *fo,
        fftwf_complex *ao);

int fftwf_clip_zero_padding(
        field_descriptor<float> *f,
        float *a);

inline float btle(const float be)
     {
         float le;
         char *befloat = (char *) & be;
         char *lefloat = (char *) & le;
         lefloat[0] = befloat[3];
         lefloat[1] = befloat[2];
         lefloat[2] = befloat[1];
         lefloat[3] = befloat[0];
         return le;
     }

#endif//FIELD_DESCRIPTOR

