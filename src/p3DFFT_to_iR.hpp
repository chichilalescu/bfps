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

#ifndef P3DFFT_TO_IR

#define P3DFFT_TO_IR

extern int myrank, nprocs;


/* this class reads in separate Fourier space p3DFFT fields
 * and generates an interleaved realspace representation,
 * using FFTW and as little memory as possible.
 * */
class p3DFFT_to_iR
{
    public:
        /* members */
        int howmany;
        field_descriptor<float> *f0c; // descriptor for 2D input
        field_descriptor<float> *f1c; // descriptor for 2D transposed input
        field_descriptor<float> *f2c; // descriptor for 3D fully transposed input
        field_descriptor<float> *f3c, *f3r; // descriptors for FFT

        fftwf_complex *c12; // array to store transposed input
        fftwf_complex *c3 ; // array to store resized Fourier data
        float *r3         ; // array to store real space data
        bool fields_allocated;

        fftwf_plan complex2real;

        /* methods */
        p3DFFT_to_iR(
                int n0, int n1, int n2,
                int N0, int N1, int N2,
                int howmany,
                bool allocate_fields = true);
        ~p3DFFT_to_iR();

        int read(
                char *ifile[]);
};

#endif//P3DFFT_TO_IR

