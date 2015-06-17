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

#include "base.hpp"
#include "p3DFFT_to_iR.hpp"
#include <string>
#include <iostream>


p3DFFT_to_iR::p3DFFT_to_iR(
        int n0, int n1, int n2,
        int N0, int N1, int N2,
        int howmany,
        bool allocate_fields)
{
    this->howmany = howmany;
    int n[7];
    DEBUG_MSG("entering constructor of p3DFFT_to_iR\n");
    if ((N0 < n2) ||
        (N1 < n1) ||
        (N2 < (n0-1)*2))
    {
        std::cerr <<
            "Output dimensions should be larger than input dimensions.\n"
            "Aborting.\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    // first 3 arguments are dimensions for input array
    // i.e. actual dimensions for the Fourier representation.
    // NOT real space grid dimensions
    // the input array is read in as a 2D array,
    // since the first dimension must be a multiple of nprocs
    // (which is generally an even number)
    n[0] = n0*n1;
    n[1] = n2;
    this->f0c = new field_descriptor<float>(2, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // f1c will be pointing at the input array after it has been
    // transposed in 2D, therefore we have this correspondence:
    // f0c->sizes[0] = f1c->sizes[1]*f1c->sizes[2]
    n[0] = n2;
    n[1] = n0;
    n[2] = n1;
    this->f1c = new field_descriptor<float>(3, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // the description for the fully transposed field
    n[0] = n2;
    n[1] = n1;
    n[2] = n0;
    this->f2c = new field_descriptor<float>(3, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // following 3 arguments are dimensions for real space grid dimensions
    // f3r and f3c will be allocated in this call
    fftwf_get_descriptors_3D(
            N0, N1, N2,
            &this->f3r, &this->f3c);

    this->fields_allocated = false;
    if (allocate_fields)
    {
        //allocate fields
        this->c12 = fftwf_alloc_complex(this->f1c->local_size);
        // 4 instead of 2, because we have 2 fields to write
        this->r3  = fftwf_alloc_real( 2*this->f3c->local_size*this->howmany);
        // all FFTs are going to be inplace
        this->c3  = (fftwf_complex*)this->r3;

        // allocate plans
        ptrdiff_t blabla[] = {this->f3r->sizes[0],
                              this->f3r->sizes[1],
                              this->f3r->sizes[2]};
        this->complex2real = fftwf_mpi_plan_many_dft_c2r(
                3,
                blabla,
                this->howmany,
                FFTW_MPI_DEFAULT_BLOCK,
                FFTW_MPI_DEFAULT_BLOCK,
                this->c3, this->r3,
                MPI_COMM_WORLD,
                FFTW_ESTIMATE);
        this->fields_allocated = true;
    }

    DEBUG_MSG("exiting constructor of p3DFFT_to_iR\n");
}

p3DFFT_to_iR::~p3DFFT_to_iR()
{
    delete this->f0c;
    delete this->f1c;
    delete this->f2c;
    delete this->f3c;
    delete this->f3r;

    if (this->fields_allocated)
    {
        fftwf_free(this->c12);
        fftwf_free(this->r3);
        fftwf_destroy_plan(this->complex2real);
    }
}

int p3DFFT_to_iR::read(
        char *ifile[])
{
    //read fields
    for (int i = 0; i < this->howmany; i++)
    {
        DEBUG_MSG("p3DFFT_to_iR::read "
                  "this->f0c->read(ifile0, ...);\n");
        this->f0c->read(
                ifile[i],
                (void*)(this->c3 + i*this->f3c->local_size));
        this->f0c->switch_endianness(
                (this->c3 + i*this->f3c->local_size));
        DEBUG_MSG("p3DFFT_to_iR::read "
                  "this->f0c->transpose(...);\n");
        this->f0c->transpose(
                this->c3 + i*this->f3c->local_size,
                this->c12);
        DEBUG_MSG("p3DFFT_to_iR::read "
                  "this->f1c->transpose(this->c12);\n");
        this->f1c->transpose(this->c12);
        DEBUG_MSG("p3DFFT_to_iR::read "
                  "fftwf_copy_complex_array(\n");
        fftwf_copy_complex_array(
                this->f2c, this->c12,
                this->f3c, this->c3 + i*this->f3c->local_size);
    }

    DEBUG_MSG("p3DFFT_to_iR::read "
              "this->f3c->interleave(this->c3, 2);\n");
    this->f3c->interleave(this->c3, this->howmany);

    DEBUG_MSG("p3DFFT_to_iR::read "
              "fftwf_execute(this->complex2real);\n");
    fftwf_execute(this->complex2real);

    DEBUG_MSG("p3DFFT_to_iR::read "
              "fftwf_clip_zero_padding(this->f3r, this->r3, 2);\n");
    fftwf_clip_zero_padding(this->f3r, this->r3, this->howmany);
    DEBUG_MSG("p3DFFT_to_iR::read return\n");
    return EXIT_SUCCESS;
}

