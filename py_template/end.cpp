    // clean up
    fftwf_mpi_cleanup();
    fftw_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

