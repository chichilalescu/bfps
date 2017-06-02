#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "scope_timer.hpp"
#include "hdf5_tools.hpp"
#include "full_code/postprocess.hpp"


int postprocess::main_loop(void)
{
    this->start_simple_timer();
    for (unsigned int iteration_counter = 0;
         iteration_counter < iteration_list.size();
         iteration_counter++)
    {
        this->iteration = iteration_list[iteration_counter];
    #ifdef USE_TIMINGOUTPUT
        const std::string loopLabel = ("postprocess::main_loop-" +
                                       std::to_string(this->iteration));
        TIMEZONE(loopLabel.c_str());
    #endif
        this->work_on_current_iteration();
        this->print_simple_timer(
                "iteration " + std::to_string(this->iteration));
    }
    return EXIT_SUCCESS;
}


int postprocess::read_parameters()
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
    dset = H5Dopen(parameter_file, "/parameters/dt", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dt);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/famplitude", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->famplitude);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/fk0", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->fk0);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/fk1", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->fk1);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/fmode", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->fmode);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/forcing_type", H5P_DEFAULT);
    space = H5Dget_space(dset);
    memtype = H5Dget_type(dset);
    string_data = (char*)malloc(256);
    H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &string_data);
    sprintf(this->forcing_type, "%s", string_data);
    free(string_data);
    H5Sclose(space);
    H5Tclose(memtype);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/nu", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nu);
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

