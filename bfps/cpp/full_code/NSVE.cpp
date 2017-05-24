#include <string>
#include <cmath>
#include "NSVE.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int NSVE<rnumber>::initialize(void)
{
    this->read_iteration();
    this->read_parameters();
    if (this->myrank == 0)
    {
        // set caching parameters
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        herr_t cache_err = H5Pset_cache(fapl, 0, 521, 134217728, 1.0);
        DEBUG_MSG("when setting stat_file cache I got %d\n", cache_err);
        this->stat_file = H5Fopen(
                (this->simname + ".h5").c_str(),
                H5F_ACC_RDWR,
                fapl);
    }
    int data_file_problem;
    if (this->myrank == 0)
        data_file_problem = this->grow_file_datasets();
    MPI_Bcast(&data_file_problem, 1, MPI_INT, 0, this->comm);
    if (data_file_problem > 0)
    {
        std::cerr <<
            data_file_problem <<
            " problems growing file datasets.\ntrying to exit now." <<
            std::endl;
        return EXIT_FAILURE;
    }
    this->fs = new vorticity_equation<rnumber, FFTW>(
            simname.c_str(),
            nx, ny, nz,
            dkx, dky, dkz,
            DEFAULT_FFTW_FLAG);
    this->tmp_vec_field = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            DEFAULT_FFTW_FLAG);


    this->fs->checkpoints_per_file = checkpoints_per_file;
    this->fs->nu = nu;
    this->fs->fmode = fmode;
    this->fs->famplitude = famplitude;
    this->fs->fk0 = fk0;
    this->fs->fk1 = fk1;
    strncpy(this->fs->forcing_type, forcing_type, 128);
    this->fs->iteration = this->iteration;
    this->fs->checkpoint = this->checkpoint;

    this->fs->cvorticity->real_space_representation = false;
    this->fs->io_checkpoint();

    if (this->myrank == 0 && this->iteration == 0)
        this->fs->kk->store(stat_file);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::step(void)
{
    this->fs->step(this->dt);
    this->iteration = this->fs->iteration;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::write_checkpoint(void)
{
    this->fs->io_checkpoint(false);
    this->checkpoint = this->fs->checkpoint;
    this->write_iteration();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::finalize(void)
{
    if (this->myrank == 0)
        H5Fclose(this->stat_file);
    delete this->fs;
    delete this->tmp_vec_field;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::do_stats()
{
    if (!(this->iteration % this->niter_stat == 0))
        return EXIT_SUCCESS;
    hid_t stat_group;
    if (this->myrank == 0)
        stat_group = H5Gopen(
                this->stat_file,
                "statistics",
                H5P_DEFAULT);
    else
        stat_group = 0;
    fs->compute_velocity(fs->cvorticity);
    *tmp_vec_field = fs->cvelocity->get_cdata();
    tmp_vec_field->compute_stats(
            fs->kk,
            stat_group,
            "velocity",
            fs->iteration / niter_stat,
            max_velocity_estimate/sqrt(3));

    *tmp_vec_field = fs->cvorticity->get_cdata();
    tmp_vec_field->compute_stats(
            fs->kk,
            stat_group,
            "vorticity",
            fs->iteration / niter_stat,
            max_vorticity_estimate/sqrt(3));

    if (this->myrank == 0)
        H5Gclose(stat_group);
    return EXIT_SUCCESS;
}

template class NSVE<float>;
template class NSVE<double>;

