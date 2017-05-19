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
    // particle sampling should go here
//    if (this->iteration % this->niter_part == 0)
//    {
//        sample_from_particles_system(*this->fs->cvelocity,// field to save TODO
//                                     this->ps,
//                                     0, // hdf5 datagroup TODO
//                                     "TODO" // dataset basename TODO
//                                     );
//    }
    return EXIT_SUCCESS;
}

template class NSVEparticles<float>;
template class NSVEparticles<double>;

