#ifndef PARTICLES_SAMPLING_HPP
#define PARTICLES_SAMPLING_HPP

#include <memory>
#include <string>

#include "abstract_particles_system.hpp"
#include "particles_output_sampling_hdf5.hpp"

#include "field.hpp"
#include "kspace.hpp"


template <class partsize_t, class particles_rnumber, class rnumber, field_backend be, field_components fc>
void sample_from_particles_system(const field<rnumber, be, fc>& in_field, // a pointer to a field<rnumber, FFTW, fc>
                                  std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>>& ps, // a pointer to an particles_system<double>
                                  const std::string& filename,
                                  const std::string& parent_groupname,
                                  const std::string& fname){
    const std::string datasetname = fname + std::string("/") + std::to_string(ps->get_step_idx());
    const int size_particle_rhs = ncomp(fc);

    // Stop here if already exists
    if(particles_output_sampling_hdf5<partsize_t, particles_rnumber, 3, size_particle_rhs>::DatasetExistsCol(MPI_COMM_WORLD,
                                                                                                             filename,
                                                                                                             parent_groupname,
                                                                                                             datasetname)){
        return;
    }

    const partsize_t nb_particles = ps->getLocalNbParticles();
    std::unique_ptr<particles_rnumber[]> sample_rhs(new particles_rnumber[size_particle_rhs*nb_particles]);
    std::fill_n(sample_rhs.get(), size_particle_rhs*nb_particles, 0);

    ps->sample_compute_field(in_field, sample_rhs.get());



    particles_output_sampling_hdf5<partsize_t, particles_rnumber, 3, size_particle_rhs> outputclass(MPI_COMM_WORLD,
                                                                                                    ps->getGlobalNbParticles(),
                                                                                                    filename,
                                                                                                    parent_groupname,
                                                                                                    datasetname);
    outputclass.save(ps->getParticlesPositions(),
                     &sample_rhs,
                     ps->getParticlesIndexes(),
                     ps->getLocalNbParticles(),
                     ps->get_step_idx());
}

template <class partsize_t, class particles_rnumber>
void sample_particles_system_position(
        std::unique_ptr<abstract_particles_system<partsize_t, particles_rnumber>>& ps, // a pointer to an particles_system<double>
                                  const std::string& filename,
                                  const std::string& parent_groupname,
                                  const std::string& fname){
    const std::string datasetname = fname + std::string("/") + std::to_string(ps->get_step_idx());

    // Stop here if already exists
    if(particles_output_sampling_hdf5<partsize_t, particles_rnumber, 3, 3>::DatasetExistsCol(MPI_COMM_WORLD,
                                                                                             filename,
                                                                                             parent_groupname,
                                                                                             datasetname)){
        return;
    }

    const partsize_t nb_particles = ps->getLocalNbParticles();
    std::unique_ptr<particles_rnumber[]> sample_rhs(new particles_rnumber[3*nb_particles]);
    std::copy(ps->getParticlesPositions(), ps->getParticlesPositions() + 3*nb_particles, sample_rhs.get());

    particles_output_sampling_hdf5<partsize_t, particles_rnumber, 3, 3> outputclass(MPI_COMM_WORLD,
                                                                                    ps->getGlobalNbParticles(),
                                                                                    filename,
                                                                                    parent_groupname,
                                                                                    datasetname);
    outputclass.save(ps->getParticlesPositions(),
                     &sample_rhs,
                     ps->getParticlesIndexes(),
                     ps->getLocalNbParticles(),
                     ps->get_step_idx());
}

#endif//PARTICLES_SAMPLING_HPP

