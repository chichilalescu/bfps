#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "scope_timer.hpp"
#include "hdf5_tools.hpp"
#include "full_code/test.hpp"


int test::main_loop(void)
{
    #ifdef USE_TIMINGOUTPUT
        TIMEZONE("test::main_loop");
    #endif
    this->start_simple_timer();
    this->do_work();
    this->print_simple_timer(
            "do_work required " + std::to_string(this->iteration));
    return EXIT_SUCCESS;
}


int test::read_parameters()
{
    hid_t parameter_file;
    hid_t dset, memtype, space;
    char fname[256];
    char *string_data;
    sprintf(fname, "%s.h5", this->simname.c_str());
    parameter_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(parameter_file, "/parameters/dealias_type", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dealias_type);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dkx", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dkx);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dky", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dky);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dkz", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dkz);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/nx", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nx);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/ny", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->ny);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/nz", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nz);
    H5Dclose(dset);
    H5Fclose(parameter_file);
    return 0;
}

