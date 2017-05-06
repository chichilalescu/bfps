#include <string>
#include <cmath>
#include "NSVEp.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int NSVEp<rnumber>::initialize(void)
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
            FFTW_MEASURE);
    this->tmp_vec_field = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            FFTW_MEASURE);


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

    this->ps = particles_system_builder(
                fs->cvelocity,              // (field object)
                fs->kk,                     // (kspace object, contains dkx, dky, dkz)
                tracers0_integration_steps, // to check coherency between parameters and hdf input file (nb rhs)
                (long long int)nparticles,  // to check coherency between parameters and hdf input file
                fs->get_current_fname(),    // particles input filename
                std::string("/tracers0/state/") + std::to_string(fs->iteration), // dataset name for initial input
                std::string("/tracers0/rhs/")  + std::to_string(fs->iteration), // dataset name for initial input
                tracers0_neighbours,        // parameter (interpolation no neighbours)
                tracers0_smoothness,        // parameter
                this->comm,
                fs->iteration+1);
    this->particles_output_writer_mpi = new particles_output_hdf5<long long int,double,3,3>(
                MPI_COMM_WORLD,
                "tracers0",
                nparticles,
                tracers0_integration_steps);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEp<rnumber>::step(void)
{
    this->fs->compute_velocity(fs->cvorticity);
    this->fs->cvelocity->ift();
    this->ps->completeLoop(this->dt);
    this->fs->step(this->dt);
    this->iteration = this->fs->iteration;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEp<rnumber>::write_checkpoint(void)
{
    this->fs->io_checkpoint(false);
    this->particles_output_writer_mpi->open_file(fs->get_current_fname());
    this->particles_output_writer_mpi->save(
            this->ps->getParticlesPositions(),
            this->ps->getParticlesRhs(),
            this->ps->getParticlesIndexes(),
            this->ps->getLocalNbParticles(),
            this->fs->iteration);
    this->particles_output_writer_mpi->close_file();
    this->checkpoint = this->fs->checkpoint;
    this->write_iteration();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEp<rnumber>::finalize(void)
{
    if (this->myrank == 0)
        H5Fclose(this->stat_file);
    delete this->fs;
    delete this->tmp_vec_field;
    this->ps.release();
    delete this->particles_output_writer_mpi;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEp<rnumber>::do_stats()
{
    // fluid stats go here
    if (this->iteration % this->niter_stat == 0)
    {
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
    }
    // particle sampling should go here
    //if (this->iteration % this->niter_part == 0)
    //{
    //}
    return EXIT_SUCCESS;
}

template class NSVEp<float>;
template class NSVEp<double>;

