/**********************************************************************
*                                                                     *
*  Copyright 2017 Max Planck Institute                                *
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



#ifndef MAIN_CODE_HPP
#define MAIN_CODE_HPP



#include <cfenv>
#include <string>
#include <iostream>
#include "base.hpp"
#include "field.hpp"
#include "scope_timer.hpp"

int myrank, nprocs;

template <class DNS>
int main_code(
        int argc,
        char *argv[],
        const bool floating_point_exceptions)
{
    /* floating point exception switch */
    if (floating_point_exceptions)
        feenableexcept(FE_INVALID | FE_OVERFLOW);
    else
        // using std::cerr because DEBUG_MSG requires myrank to be defined
        std::cerr << "FPE have been turned OFF" << std::endl;

    if (argc != 2)
    {
        std::cerr <<
            "Wrong number of command line arguments. Stopping." <<
            std::endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    std::string simname = std::string(argv[1]);


    /* initialize MPI environment */
#ifdef NO_FFTWOMP
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    fftw_mpi_init();
    fftwf_mpi_init();
    DEBUG_MSG("There are %d processes\n", nprocs);
#else
    int mpiprovided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiprovided);
    assert(mpiprovided >= MPI_THREAD_FUNNELED);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int nThreads = omp_get_max_threads();
    DEBUG_MSG("Number of threads for the FFTW = %d\n",
              nThreads);
    if (nThreads > 1){
        fftw_init_threads();
        fftwf_init_threads();
    }
    fftw_mpi_init();
    fftwf_mpi_init();
    DEBUG_MSG("There are %d processes and %d threads\n",
              nprocs,
              nThreads);
    if (nThreads > 1){
        fftw_plan_with_nthreads(nThreads);
        fftwf_plan_with_nthreads(nThreads);
    }
#endif



    /* import fftw wisdom */
    if (myrank == 0)
        fftwf_import_wisdom_from_filename(
                (simname + std::string("_fftw_wisdom.txt")).c_str());
    fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);



    /* actually run DNS */
    /*
     * MPI environment:
     * I could in principle pass myrank and nprocs instead of the global
     * communicator, but it is possible that we'd like to do something more
     * complex in the future (since I've done it in the past), and it's not
     * expensive to keep several copies of myrank and nprocs.
     *
     * usage of assert:
     * we could use assert here, but I assume that any problems we can still
     * recover from should not be important enough to not clean up fftw and MPI
     * things.
     */
    DNS *dns = new DNS(
            MPI_COMM_WORLD,
            simname);
    int return_value;
    return_value = dns->initialize();
    if (return_value == EXIT_SUCCESS)
        return_value = dns->main_loop();
    else
        DEBUG_MSG("problem calling dns->initialize(), return value is %d",
                  return_value);
    if (return_value == EXIT_SUCCESS)
        return_value = dns->finalize();
    else
        DEBUG_MSG("problem calling dns->main_loop(), return value is %d",
                  return_value);
    if (return_value != EXIT_SUCCESS)
        DEBUG_MSG("problem calling dns->finalize(), return value is %d",
                  return_value);

    delete dns;



    /* export fftw wisdom */
    fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0)
        fftwf_export_wisdom_to_filename(
                (simname + std::string("_fftw_wisdom.txt")).c_str());



    /* clean up */
    fftwf_mpi_cleanup();
    fftw_mpi_cleanup();
#ifndef NO_FFTWOMP
    if (nThreads > 1){
        fftw_cleanup_threads();
        fftwf_cleanup_threads();
    }
#endif
#ifdef USE_TIMINGOUTPUT
    global_timer_manager.show(MPI_COMM_WORLD);
    global_timer_manager.showHtml(MPI_COMM_WORLD);
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}


#endif//MAIN_CODE_HPP

