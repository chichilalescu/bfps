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

#include "p3DFFT_to_iR.hpp"
#include "Morton_shuffler.hpp"
#include <iostream>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n, N, nfiles, nfields;
    int test_type = 0;
    switch(argc)
    {
        case 6:
            test_type = atoi(argv[5]);
        case 5:
            n = atoi(argv[1]);
            N = atoi(argv[2]);
            nfiles = atoi(argv[3]);
            nfields = atoi(argv[4]);
            break;
        default:
            std::cerr <<
                "not enough (or too many) parameters.\naborting." <<
                std::endl;
            MPI_Finalize();
            return EXIT_SUCCESS;
    }
    p3DFFT_to_iR *r = new p3DFFT_to_iR(
            (n/2+1), n, n,
            N, N, N,
            nfields,
            test_type);

    Morton_shuffler *s = new Morton_shuffler(
            N, N, N, nfields, nfiles);

    if (test_type)
    {
        // initialize file names
        char* ifile[nfields];
        for (int i=0; i<nfields; i++)
        {
            ifile[i] = (char*)malloc(100*sizeof(char));
            sprintf(ifile[i], "Kdata%d", i);
        }

        //read
        r->read(ifile);
        for (int i = 0; i<nfields; i++)
            free(ifile[i]);
        s->shuffle(r->r3, "Rdata");

        delete s;
    }
    delete r;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

