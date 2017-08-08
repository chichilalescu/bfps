#include <string>
#include <cmath>
#include "joint_acc_vel_stats.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int joint_acc_vel_stats<rnumber>::initialize(void)
{
    this->NSVE_field_stats<rnumber>::initialize();
    this->kk = new kspace<FFTW, SMOOTH>(
            this->vorticity->clayout, this->dkx, this->dky, this->dkz);
    this->ve = new vorticity_equation<rnumber, FFTW>(
            this->simname.c_str(),
            this->nx,
            this->ny,
            this->nz,
            this->dkx,
            this->dky,
            this->dkz,
            this->vorticity->fftw_plan_rigor);
    hid_t parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    hid_t dset = H5Dopen(parameter_file, "/parameters/niter_out", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->niter_out);
    H5Dclose(dset);
    if (H5Lexists(parameter_file, "/parameters/checkpoints_per_file", H5P_DEFAULT))
    {
        dset = H5Dopen(parameter_file, "/parameters/checkpoints_per_file", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->checkpoints_per_file);
        H5Dclose(dset);
    }
    else
        this->checkpoints_per_file = 1;
    H5Fclose(parameter_file);
    parameter_file = H5Fopen(
            (this->simname + std::string("_post.h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    this->iteration_list = hdf5_tools::read_vector<int>(
            parameter_file,
            "/joint_acc_vel_stats/parameters/iteration_list");
    dset = H5Dopen(parameter_file, "joint_acc_vel_stats/parameters/max_acceleration_estimate", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->max_acceleration_estimate);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "joint_acc_vel_stats/parameters/max_velocity_estimate", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->max_velocity_estimate);
    H5Dclose(dset);
    H5Fclose(parameter_file);
    if (this->myrank == 0)
    {
        // set caching parameters
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        herr_t cache_err = H5Pset_cache(fapl, 0, 521, 134217728, 1.0);
        DEBUG_MSG("when setting stat_file cache I got %d\n", cache_err);
        this->stat_file = H5Fopen(
                (this->simname + "_post.h5").c_str(),
                H5F_ACC_RDWR,
                fapl);
    }
    else
    {
        this->stat_file = 0;
    }
    int data_file_problem;
    if (this->myrank == 0)
        data_file_problem = hdf5_tools::require_size_file_datasets(
                this->stat_file,
                "joint_acc_vel_stats",
                (this->iteration_list.back() / this->niter_out) + 1);
    MPI_Bcast(&data_file_problem, 1, MPI_INT, 0, this->comm);
    if (data_file_problem > 0)
    {
        std::cerr <<
            data_file_problem <<
            " problems setting sizes of file datasets.\ntrying to exit now." <<
            std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

template <typename rnumber>
int joint_acc_vel_stats<rnumber>::work_on_current_iteration(void)
{
    DEBUG_MSG("entered joint_acc_vel_stats::work_on_current_iteration\n");
    /// read current vorticity, place it in this->ve->cvorticity
    this->read_current_cvorticity();
    *this->ve->cvorticity = this->vorticity->get_cdata();
    /// after the previous instruction, we are free to use this->vorticity
    /// for any other purpose

    /// initialize `stat_group`.
    hid_t stat_group;
    if (this->myrank == 0)
        stat_group = H5Gopen(
                this->stat_file,
                "joint_acc_vel_stats",
                H5P_DEFAULT);
    else
        stat_group = 0;

    field<rnumber, FFTW, THREE> *vel;
    field<rnumber, FFTW, THREE> *acc;

    /// compute velocity
    vel = new field<rnumber, FFTW, THREE>(
            this->nx, this->ny, this->nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    invert_curl(kk, this->ve->cvorticity, vel);
    vel->ift();

    /// compute Lagrangian acceleration
    /// use this->vorticity as temporary field
    acc = this->vorticity;
    this->ve->compute_Lagrangian_acceleration(acc);
    acc->ift();

    /// compute single field real space statistics
    std::vector<double> max_acc_estimate;
    std::vector<double> max_vel_estimate;

    max_acc_estimate.resize(4, max_acceleration_estimate / sqrt(3));
    max_vel_estimate.resize(4, max_velocity_estimate / sqrt(3));
    max_acc_estimate[3] = max_acceleration_estimate;
    max_vel_estimate[3] = max_velocity_estimate;

    acc->compute_rspace_stats(
            stat_group,
            "acceleration",
            this->iteration / this->niter_out,
            max_acc_estimate);
    vel->compute_rspace_stats(
            stat_group,
            "velocity",
            this->iteration / this->niter_out,
            max_vel_estimate);

    /// compute joint PDF
    joint_rspace_PDF(
            acc, vel,
            stat_group,
            "acceleration_and_velocity",
            this->iteration / this->niter_out,
            max_acc_estimate,
            max_vel_estimate);

    delete vel;

    return EXIT_SUCCESS;
}

template <typename rnumber>
int joint_acc_vel_stats<rnumber>::finalize(void)
{
    delete this->ve;
    delete this->kk;
    if (this->myrank == 0)
        H5Fclose(this->stat_file);
    this->NSVE_field_stats<rnumber>::finalize();
    return EXIT_SUCCESS;
}

template class joint_acc_vel_stats<float>;
template class joint_acc_vel_stats<double>;

