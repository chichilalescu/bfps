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
    MPI_Comm mpi_com;

    int my_rank;
    int nb_processes;

    const int total_nb_particles;
    const int nb_rhs;

    std::unique_ptr<std::pair<int,int>[]> buffer_indexes_send;
    std::unique_ptr<real_number[]> buffer_particles_positions_send;
    std::vector<std::unique_ptr<real_number[]>> buffer_particles_rhs_send;
    int size_buffers_send;

    std::unique_ptr<real_number[]> buffer_particles_positions_recv;
    std::vector<std::unique_ptr<real_number[]>> buffer_particles_rhs_recv;
    int size_buffers_recv;


protected:
    MPI_Comm& getCom(){
        return mpi_com;
    }

    int getTotalNbParticles() const {
        return total_nb_particles;
    }

    int getNbRhs() const {
        return nb_rhs;
    }

public:
    abstract_particles_output(MPI_Comm in_mpi_com, const int inTotalNbParticles, const int in_nb_rhs)
            : mpi_com(in_mpi_com), my_rank(-1), nb_processes(-1),
                total_nb_particles(inTotalNbParticles), nb_rhs(in_nb_rhs),
                buffer_particles_rhs_send(in_nb_rhs), size_buffers_send(-1),
                buffer_particles_rhs_recv(in_nb_rhs), size_buffers_recv(-1){

        AssertMpi(MPI_Comm_rank(mpi_com, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_com, &nb_processes));
    }

    virtual ~abstract_particles_output(){
    }

    void releaseMemory(){
        buffer_indexes_send.release();
        buffer_particles_positions_send.release();
        size_buffers_send = -1;
        buffer_particles_positions_recv.release();
        size_buffers_recv = -1;
        for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
            buffer_particles_rhs_send[idx_rhs].release();
            buffer_particles_rhs_recv[idx_rhs].release();
        }
    }

    void save(const real_number input_particles_positions[], const std::unique_ptr<real_number[]> input_particles_rhs[],
              const int index_particles[], const int nb_particles, const int idx_time_step){
        TIMEZONE("abstract_particles_output::save");
        assert(total_nb_particles != -1);
        DEBUG_MSG("[%d] total_nb_particles %d \n", my_rank, total_nb_particles);
        DEBUG_MSG("[%d] nb_particles %d to distribute for saving \n", my_rank, nb_particles);

        {
            TIMEZONE("sort");

            if(size_buffers_send < nb_particles && nb_particles){
                buffer_indexes_send.reset(new std::pair<int,int>[nb_particles]);
                buffer_particles_positions_send.reset(new real_number[nb_particles*size_particle_positions]);
                for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                    buffer_particles_rhs_send[idx_rhs].reset(new real_number[nb_particles*size_particle_rhs]);
                }
                size_buffers_send = nb_particles;
            }

            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                buffer_indexes_send[idx_part].first = idx_part;
                buffer_indexes_send[idx_part].second = index_particles[idx_part];
            }

            std::sort(&buffer_indexes_send[0], &buffer_indexes_send[nb_particles], [](const std::pair<int,int>& p1, const std::pair<int,int>& p2){
                return p1.second < p2.second;
            });

            for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                const int src_idx = buffer_indexes_send[idx_part].first;
                const int dst_idx = idx_part;

                for(int idx_val = 0 ; idx_val < size_particle_positions ; ++idx_val){
                    buffer_particles_positions_send[dst_idx*size_particle_positions + idx_val]
                            = input_particles_positions[src_idx*size_particle_positions + idx_val];
                }
                for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                    for(int idx_val = 0 ; idx_val < int(size_particle_rhs) ; ++idx_val){
                        buffer_particles_rhs_send[idx_rhs][dst_idx*size_particle_rhs + idx_val]
                                = input_particles_rhs[idx_rhs][src_idx*size_particle_rhs + idx_val];
                    }
                }
            }
        }

        const particles_utils::IntervalSplitter<int> particles_splitter(total_nb_particles, nb_processes, my_rank);
        DEBUG_MSG("[%d] nb_particles_per_proc %d for saving\n", my_rank, particles_splitter.getMySize());

        std::vector<int> nb_particles_to_send(nb_processes, 0);
        for(int idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
            nb_particles_to_send[particles_splitter.getOwner(buffer_indexes_send[idx_part].second)] += 1;
        }

        alltoall_exchanger exchanger(mpi_com, std::move(nb_particles_to_send));
        // nb_particles_to_send is invalid after here

        const int nb_to_receive = exchanger.getTotalToRecv();

        if(size_buffers_recv < nb_to_receive && nb_to_receive){
            buffer_particles_positions_recv.reset(new real_number[nb_to_receive*size_particle_positions]);
            for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                buffer_particles_rhs_recv[idx_rhs].reset(new real_number[nb_to_receive*size_particle_rhs]);
            }
            size_buffers_recv = nb_to_receive;
        }

        // Could be done with multiple asynchronous coms
        exchanger.alltoallv(buffer_particles_positions_send.get(), buffer_particles_positions_recv.get(), size_particle_positions);
        for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
            exchanger.alltoallv(buffer_particles_rhs_send[idx_rhs].get(), buffer_particles_rhs_recv[idx_rhs].get(), size_particle_rhs);
        }

        write(idx_time_step, buffer_particles_positions_recv.get(), buffer_particles_rhs_recv.data(),
              nb_to_receive, particles_splitter.getMyOffset());
    }

    virtual void write(const int idx_time_step, const real_number* positions, const std::unique_ptr<real_number[]>* rhs,
                       const int nb_particles, const int particles_idx_offset) = 0;
};

#endif
