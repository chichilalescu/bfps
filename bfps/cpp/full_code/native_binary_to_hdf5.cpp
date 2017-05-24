#include <string>
#include <cmath>
#include "native_binary_to_hdf5.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int native_binary_to_hdf5<rnumber>::initialize(void)
{
    this->read_parameters();
    this->vec_field = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    this->vec_field->real_space_representation = false;
    this->bin_IO = new field_binary_IO<rnumber, COMPLEX, THREE>(
            this->vec_field->clayout->sizes,
            this->vec_field->clayout->subsizes,
            this->vec_field->clayout->starts,
            this->vec_field->clayout->comm);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int native_binary_to_hdf5<rnumber>::work_on_current_iteration(void)
{
    char itername[16];
    sprintf(itername, "i%.5x", this->iteration);
    std::string native_binary_fname = (
            this->simname +
            std::string("_cvorticity_") +
            std::string(itername));
    this->bin_IO->read(
            native_binary_fname,
            this->vec_field->get_cdata());
    this->vec_field->io(
            (native_binary_fname +
             std::string(".h5")),
            "vorticity",
            this->iteration,
            false);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int native_binary_to_hdf5<rnumber>::finalize(void)
{
    delete this->bin_IO;
    delete this->vec_field;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int native_binary_to_hdf5<rnumber>::read_parameters(void)
{
    this->postprocess::read_parameters();
    hid_t parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    this->iteration_list = hdf5_tools::read_vector<int>(
            parameter_file,
            "/native_binary_to_hdf5/iteration_list");
    return EXIT_SUCCESS;
}

template class native_binary_to_hdf5<float>;
template class native_binary_to_hdf5<double>;

