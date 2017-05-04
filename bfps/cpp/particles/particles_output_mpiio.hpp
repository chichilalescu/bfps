#ifndef PARTICLES_OUTPUT_MPIIO
#define PARTICLES_OUTPUT_MPIIO

#include <memory>
#include <vector>
#include <string>
#include <cassert>

#include "abstract_particles_output.hpp"
#include "scope_timer.hpp"
#include "particles_utils.hpp"

template <class real_number, int size_particle_positions, int size_particle_rhs>
class particles_output_mpiio : public abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>{
    using Parent = abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>;

    const std::string filename;
    const int nb_step_prealloc;

    int current_step_in_file;

    MPI_File mpi_file;

public:
    particles_output_mpiio(MPI_Comm in_mpi_com, const std::string in_filename, const int inTotalNbParticles,
                           const int in_nb_rhs, const int in_nb_step_prealloc = -1)
            : abstract_particles_output<real_number, size_particle_positions, size_particle_rhs>(in_mpi_com, inTotalNbParticles, in_nb_rhs),
              filename(in_filename), nb_step_prealloc(in_nb_step_prealloc), current_step_in_file(0){
        if(Parent::isInvolved()){
            {
                TIMEZONE("particles_output_mpiio::MPI_File_open");
                AssertMpi(MPI_File_open(Parent::getComWriter(), const_cast<char*>(filename.c_str()),
                    MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file));
            }
            if(nb_step_prealloc != -1){
                TIMEZONE("particles_output_mpiio::MPI_File_set_size");
                AssertMpi(MPI_File_set_size(mpi_file,
                    nb_step_prealloc*Parent::getTotalNbParticles()*sizeof(real_number)*(size_particle_positions+size_particle_rhs*Parent::getNbRhs())));
            }
        }
    }

    ~particles_output_mpiio(){
        if(Parent::isInvolved()){
            TIMEZONE("particles_output_mpiio::MPI_File_close");
            AssertMpi(MPI_File_close(&mpi_file));
        }
    }

    void write(const int /*time_step*/, const real_number* particles_positions, const std::unique_ptr<real_number[]>* particles_rhs,
                           const int nb_particles, const int particles_idx_offset) final{
        assert(Parent::isInvolved());

        TIMEZONE("particles_output_mpiio::write");

        assert(nb_step_prealloc == -1 || current_step_in_file < nb_step_prealloc);
        assert(particles_idx_offset < Parent::getTotalNbParticles());
        assert(particles_idx_offset+nb_particles <= Parent::getTotalNbParticles());

        if(nb_step_prealloc == -1){
            TIMEZONE("particles_output_mpiio::write::MPI_File_set_size");
            AssertMpi(MPI_File_set_size(mpi_file,
                (current_step_in_file+1)*Parent::getTotalNbParticles()*sizeof(real_number)*(size_particle_positions+size_particle_rhs*Parent::getNbRhs())));
        }

        const MPI_Offset globalParticlesOffset = current_step_in_file*Parent::getTotalNbParticles()*(size_particle_positions+size_particle_rhs*Parent::getNbRhs())
                        + nb_particles*size_particle_positions;

        const MPI_Offset writingOffset = globalParticlesOffset * sizeof(real_number);

        AssertMpi(MPI_File_write_at(mpi_file, writingOffset,
            const_cast<real_number*>(particles_positions), nb_particles*size_particle_positions, particles_utils::GetMpiType(real_number()),
            MPI_STATUS_IGNORE));

        for(int idx_rsh = 0 ; idx_rsh < Parent::getNbRhs() ; ++idx_rsh){
            const MPI_Offset globalParticlesOffsetOutput = current_step_in_file*Parent::getTotalNbParticles()*(size_particle_positions+size_particle_rhs)
                            + Parent::getTotalNbParticles()*size_particle_positions
                            + idx_rsh*Parent::getTotalNbParticles()*size_particle_rhs
                            + nb_particles*size_particle_rhs;

            const MPI_Offset writingOffsetOutput = globalParticlesOffsetOutput * sizeof(real_number);

            AssertMpi(MPI_File_write_at(mpi_file, writingOffsetOutput,
                const_cast<real_number*>(particles_rhs[idx_rsh].get()), nb_particles*size_particle_rhs, particles_utils::GetMpiType(real_number()),
                MPI_STATUS_IGNORE));
        }

        current_step_in_file += 1;
    }
};

#endif
