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
                                  hid_t gid, // an hid_t  identifying an HDF5 group
                                  const std::string& fname,
                                  const partsize_t totalNbParticles){
//    const int size_particle_rhs = ncomp(fc);
//    const partsize_t nb_particles = ps->getLocalNbParticles();
//    std::unique_ptr<particles_rnumber[]> sample_rhs(new particles_rnumber[size_particle_rhs*nb_particles]);

//    ps->sample_compute_field(in_field, sample_rhs.get());

//    const std::string datasetname = fname + std::string("/") + std::to_string(ps->step_idx);

//    particles_output_sampling_hdf5<partsize_t, particles_rnumber, 3, size_particle_rhs> outputclass(MPI_COMM_WORLD,
//                                                                                                    totalNbParticles,
//                                                                                                    gid,
//                                                                                                    datasetname);
//    outputclass.save(ps->getParticlesPositions(),
//                     &sample_rhs,
//                     ps->getParticlesIndexes(),
//                     ps->getLocalNbParticles(),
//                     in_field->iteration);
}

#endif

