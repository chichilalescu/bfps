from base import base


class code(base):
    def __init__(self):
        self.main_start = """
                //@begincpp
                #include "base.hpp"
                #include "fluid_solver.hpp"
                #include <iostream>
                #include <fftw3-mpi.h>

                int myrank, nprocs;

                int main(int argc, char *argv[])
                {
                    MPI_Init(&argc, &argv);
                    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
                //@endcpp"""
        self.main_end = """
                //@begincpp
                    // clean up
                    fftwf_mpi_cleanup();
                    fftw_mpi_cleanup();
                    MPI_Finalize();
                    return EXIT_SUCCESS;
                }
                //@endcpp"""
        return None

