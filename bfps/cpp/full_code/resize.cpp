#include <string>
#include <cmath>
#include "resize.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int resize<rnumber>::initialize(void)
{
    this->NSVE_field_stats<rnumber>::initialize();
    DEBUG_MSG("after NSVE_field_stats::initialize\n");
    hid_t parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);

    this->niter_out = hdf5_tools::read_value<int>(
            parameter_file, "/parameters/niter_out");
    H5Fclose(parameter_file);
    parameter_file = H5Fopen(
            (this->simname + std::string("_post.h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    DEBUG_MSG("before read_vector\n");
    this->iteration_list = hdf5_tools::read_vector<int>(
            parameter_file,
            "/resize/parameters/iteration_list");

    this->new_nx = hdf5_tools::read_value<int>(
            parameter_file, "/resize/parameters/new_nx");
    this->new_ny = hdf5_tools::read_value<int>(
            parameter_file, "/resize/parameters/new_ny");
    this->new_nz = hdf5_tools::read_value<int>(
            parameter_file, "/resize/parameters/new_nz");
    this->new_simname = hdf5_tools::read_string(
            parameter_file, "/resize/parameters/new_simname");
    H5Fclose(parameter_file);

    this->new_field = new field<rnumber, FFTW, THREE>(
            this->new_nx, this->new_ny, this->new_nz,
            this->comm,
            this->vorticity->fftw_plan_rigor);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int resize<rnumber>::work_on_current_iteration(void)
{
    DEBUG_MSG("entered resize::work_on_current_iteration\n");
    this->read_current_cvorticity();

    std::string fname = (
            this->new_simname +
            std::string("_fields.h5"));
    this->new_field = this->vorticity;
    this->new_field->io(
            fname,
            "vorticity",
            this->iteration,
            false);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int resize<rnumber>::finalize(void)
{
    delete this->new_field;
    this->NSVE_field_stats<rnumber>::finalize();
    return EXIT_SUCCESS;
}

template class resize<float>;
template class resize<double>;

