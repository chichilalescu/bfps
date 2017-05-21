#include <string>
#include <cmath>
#include "NSVEparticles.hpp"
#include "scope_timer.hpp"
#include "particles/particles_sampling.hpp"

template <typename rnumber>
int NSVEparticles<rnumber>::initialize(void)
{
    this->NSVE<rnumber>::initialize();

    this->ps = particles_system_builder(
                this->fs->cvelocity,              // (field object)
                this->fs->kk,                     // (kspace object, contains dkx, dky, dkz)
                tracers0_integration_steps, // to check coherency between parameters and hdf input file (nb rhs)
                (long long int)nparticles,  // to check coherency between parameters and hdf input file
                this->fs->get_current_fname(),    // particles input filename
                std::string("/tracers0/state/") + std::to_string(this->fs->iteration), // dataset name for initial input
                std::string("/tracers0/rhs/")  + std::to_string(this->fs->iteration), // dataset name for initial input
                tracers0_neighbours,        // parameter (interpolation no neighbours)
                tracers0_smoothness,        // parameter
                this->comm,
                this->fs->iteration+1);
    this->particles_output_writer_mpi = new particles_output_hdf5<long long int,double,3,3>(
                MPI_COMM_WORLD,
                "tracers0",
                nparticles,
                tracers0_integration_steps);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::step(void)
{
    this->fs->compute_velocity(this->fs->cvorticity);
    this->fs->cvelocity->ift();
    this->ps->completeLoop(this->dt);
    this->NSVE<rnumber>::step();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::write_checkpoint(void)
{
    this->NSVE<rnumber>::write_checkpoint();
    this->particles_output_writer_mpi->open_file(this->fs->get_current_fname());
    this->particles_output_writer_mpi->save(
            this->ps->getParticlesPositions(),
            this->ps->getParticlesRhs(),
            this->ps->getParticlesIndexes(),
            this->ps->getLocalNbParticles(),
            this->fs->iteration);
    this->particles_output_writer_mpi->close_file();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::finalize(void)
{
    this->NSVE<rnumber>::finalize();
    this->ps.release();
    delete this->particles_output_writer_mpi;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::do_stats()
{
    // fluid stats go here
    this->NSVE<rnumber>::do_stats();


    if (!(this->iteration % this->niter_part == 0))
        return EXIT_SUCCESS;

    hid_t file_id, pgroup_id;

    // open particle file, and tracers0 group
    hid_t plist_id_par = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id_par >= 0);
    int retTest = H5Pset_fapl_mpio(
            plist_id_par,
            this->particles_output_writer_mpi->getComWriter(),
            MPI_INFO_NULL);
    assert(retTest >= 0);

    // Parallel HDF5 write
    file_id = H5Fopen(
            (this->simname + "_particles.h5").c_str(),
            H5F_ACC_RDWR | H5F_ACC_DEBUG,
            plist_id_par);
    // file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC | H5F_ACC_DEBUG/*H5F_ACC_EXCL*/, H5P_DEFAULT/*H5F_ACC_RDWR*/, plist_id_par);
    assert(file_id >= 0);
    H5Pclose(plist_id_par);

    pgroup_id = H5Gopen(
            file_id,
            "tracers0",
            H5P_DEFAULT);

    //after fluid stats, cvelocity contains Fourier representation of vel field
    this->fs->cvelocity->ift();


    // sample velocity
    sample_from_particles_system(*this->fs->cvelocity,// field to save
                                 this->ps,
                                 pgroup_id, // hdf5 datagroup TODO
                                 "velocity" // dataset basename TODO
                                 );
    H5Gclose(pgroup_id);
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}

template class NSVEparticles<float>;
template class NSVEparticles<double>;

