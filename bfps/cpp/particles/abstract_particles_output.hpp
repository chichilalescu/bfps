#ifndef ABSTRACT_PARTICLES_OUTPUT
#define ABSTRACT_PARTICLES_OUTPUT

#include <memory>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>

#include "base.hpp"
#include "particles_utils.hpp"
#include "alltoall_exchanger.hpp"
#include "scope_timer.hpp"


template <class real_number, int size_particle_positions, int size_particle_rhs>
class abstract_particles_output {
    struct movable_particle{
        int global_idx;
        real_number positions[size_particle_positions];
        real_number rhs[size_particle_rhs];
    };

    void create_movable_mpi_data_type(){
        /** Type in order in the struct */
        MPI_Datatype type[3] = { MPI_INT,
                                 particles_utils::GetMpiType(real_number()),
                                 particles_utils::GetMpiType(real_number()) };
        /** Number of occurence of each type */
        int blocklen[3] = { 1, size_particle_positions, size_particle_rhs };
        /** Position offset from struct starting address */
        MPI_Aint disp[3];
        disp[0] = offsetof(movable_particle,global_idx);
        disp[1] = offsetof(movable_particle,positions);
        disp[2] = offsetof(movable_particle,rhs);
        /** Create the type */
        AssertMpi( MPI_Type_create_struct(3, blocklen, disp, type, &mpi_movable_particle_type) );
        /** Commit it*/
        AssertMpi( MPI_Type_commit(&mpi_movable_particle_type) );
    }

    MPI_Datatype mpi_movable_particle_type;

    MPI_Comm mpi_com;

    int my_rank;
    int nb_processes;

    std::unique_ptr<movable_particle[]> buffer_particles_send;
    int nb_particles_allocated_send;
    const int total_nb_particles;

    std::unique_ptr<movable_particle[]> buffer_particles_recv;
    int nb_particles_allocated_recv;

protected:
    MPI_Comm& getCom(){
        return mpi_com;
    }

    int getTotalNbParticles() const {
        return total_nb_particles;
    }

public:
    abstract_particles_output(MPI_Comm in_mpi_com, const int inTotalNbParticles)
            : mpi_com(in_mpi_com), my_rank(-1), nb_processes(-1),
                nb_particles_allocated_send(-1), total_nb_particles(inTotalNbParticles),
                nb_particles_allocated_recv(-1){

        AssertMpi(MPI_Comm_rank(mpi_com, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_com, &nb_processes));

        create_movable_mpi_data_type();
    }

    virtual ~abstract_particles_output(){
        MPI_Type_free(&mpi_movable_particle_type);
    }

    void releaseMemory(){
        buffer_particles_send.release();
        nb_particles_allocated_send = -1;
        buffer_particles_recv.release();
        nb_particles_allocated_recv = -1;
    }

    void save(const real_number input_particles_positions[], const real_number input_particles_rhs[],
              const int index_particles[], const int nb_particles, const int idx_time_step){
        TIMEZONE("abstract_particles_output::save");
        assert(total_nb_particles != -1);
        DEBUG_MSG("[%d] total_nb_particles %d \n", my_rank, total_nb_particles);
        DEBUG_MSG("[%d] nb_particles %d to distribute for saving \n", my_rank, nb_particles);

        {
            TIMEZONE("sort");

            if(nb_particles_allocated_send < nb_particles){
                buffer_particles_send.reset(new movable_particle[nb_particles]);
                nb_particles_allocated_send = nb_particles;
            }

            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                buffer_particles_send[idx_part].global_idx = index_particles[idx_part];
                for(int idx_val = 0 ; idx_val < size_particle_positions ; ++idx_val){
                    buffer_particles_send[idx_part].positions[idx_val] = input_particles_positions[idx_part*size_particle_positions + idx_val];
                }
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    buffer_particles_send[idx_part].rhs[idx_val] = input_particles_rhs[idx_part*size_particle_rhs + idx_val];
                }
            }

            std::sort(&buffer_particles_send[0], &buffer_particles_send[nb_particles], [](const movable_particle& p1, const movable_particle& p2){
                return p1.global_idx < p2.global_idx;
            });
        }

        const particles_utils::IntervalSplitter<int> particles_splitter(total_nb_particles, nb_processes, my_rank);
        DEBUG_MSG("[%d] nb_particles_per_proc %d for saving\n", my_rank, particles_splitter.getMySize());

        std::vector<int> nb_particles_to_send(nb_processes, 0);
        for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
            nb_particles_to_send[particles_splitter.getOwner(buffer_particles_send[idx_part].global_idx)] += 1;
        }

        alltoall_exchanger exchanger(mpi_com, std::move(nb_particles_to_send));
        // nb_particles_to_send is invalid after here

        const int nb_to_receive = exchanger.getTotalToRecv();

        if(nb_particles_allocated_recv < nb_to_receive){
            buffer_particles_recv.reset(new movable_particle[nb_to_receive]);
            nb_particles_allocated_recv = nb_to_receive;
        }

        exchanger.alltoallv_dt(buffer_particles_send.get(), buffer_particles_recv.get(), mpi_movable_particle_type);

        // Trick re-use the buffer to have only real_number

        if(nb_particles_allocated_send < nb_to_receive){
            buffer_particles_send.reset(new movable_particle[nb_to_receive]);
            nb_particles_allocated_send = nb_to_receive;
        }

        real_number* buffer_positions_dptr = reinterpret_cast<real_number*>(buffer_particles_recv.get());
        real_number* buffer_rhs_dptr = reinterpret_cast<real_number*>(buffer_particles_send.get());
        {
            TIMEZONE("copy");
            for(int idx_part = 0 ; idx_part < nb_to_receive ; ++idx_part){
                for(int idx_val = 0 ; idx_val < size_particle_positions ; ++idx_val){
                    buffer_positions_dptr[idx_part*size_particle_positions + idx_val]
                            = buffer_particles_recv[idx_part].positions[idx_val];
                }
                // Can be done here or before "positions" copy
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    buffer_rhs_dptr[idx_part*size_particle_rhs + idx_val]
                            = buffer_particles_recv[idx_part].rhs[idx_val];
                }
            }
        }

        write(idx_time_step, buffer_positions_dptr, buffer_rhs_dptr, nb_to_receive, particles_splitter.getMyOffset());
    }

    virtual void write(const int idx_time_step, const real_number* positions, const real_number* rhs,
                       const int nb_particles, const int particles_idx_offset) = 0;
};

#endif
