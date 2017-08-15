#include <string>
#include <cmath>
#include "NSVE_field_stats.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int NSVE_field_stats<rnumber>::initialize(void)
{
    this->postprocess::read_parameters();
    this->vorticity = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    this->vorticity->real_space_representation = false;
    hid_t parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    if (!H5Lexists(parameter_file, "field_dtype", H5P_DEFAULT))
        this->bin_IO = NULL;
    else
    {
        hid_t dset = H5Dopen(parameter_file, "field_dtype", H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hid_t memtype = H5Dget_type(dset);
        char *string_data = (char*)malloc(256);
        H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &string_data);
        // check that we're using the correct data type
        // field_dtype SHOULD be something like "<f4", "<f8", ">f4", ">f8"
        // first character is ordering, which is machine specific
        // for the other two I am checking that they have the correct values
        assert(string_data[1] == 'f');
        assert(string_data[2] == '0' + sizeof(rnumber));
        free(string_data);
        H5Sclose(space);
        H5Tclose(memtype);
        H5Dclose(dset);
        this->bin_IO = new field_binary_IO<rnumber, COMPLEX, THREE>(
                this->vorticity->clayout->sizes,
                this->vorticity->clayout->subsizes,
                this->vorticity->clayout->starts,
                this->vorticity->clayout->comm);
    }
    H5Fclose(parameter_file);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE_field_stats<rnumber>::read_current_cvorticity(void)
{
    this->vorticity->real_space_representation = false;
    if (this->bin_IO != NULL)
    {
        char itername[16];
        sprintf(itername, "i%.5x", this->iteration);
        std::string native_binary_fname = (
                this->simname +
                std::string("_cvorticity_") +
                std::string(itername));
        this->bin_IO->read(
                native_binary_fname,
                this->vorticity->get_cdata());
    }
    else
    {
        this->vorticity->io(
                this->simname + std::string("_fields.h5"),
                "vorticity",
                this->iteration,
                true);
    }
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE_field_stats<rnumber>::finalize(void)
{
    if (this->bin_IO != NULL)
        delete this->bin_IO;
    delete this->vorticity;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE_field_stats<rnumber>::work_on_current_iteration(void)
{
    return EXIT_SUCCESS;
}

template class NSVE_field_stats<float>;
template class NSVE_field_stats<double>;

