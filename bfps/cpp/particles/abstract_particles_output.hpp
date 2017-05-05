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
#include "env_utils.hpp"

template <class partsize_t, class real_number, int size_particle_positions, int size_particle_rhs>
class abstract_particles_output {
    MPI_Comm mpi_com;
    MPI_Comm mpi_com_writer;

    int my_rank;
    int nb_processes;

    const partsize_t total_nb_particles;
    const int nb_rhs;

    std::unique_ptr<std::pair<partsize_t,partsize_t>[]> buffer_indexes_send;
    std::unique_ptr<real_number[]> buffer_particles_positions_send;
    std::vector<std::unique_ptr<real_number[]>> buffer_particles_rhs_send;
    partsize_t size_buffers_send;

    std::unique_ptr<real_number[]> buffer_particles_positions_recv;
    std::vector<std::unique_ptr<real_number[]>> buffer_particles_rhs_recv;
    std::unique_ptr<partsize_t[]> buffer_indexes_recv;
    partsize_t size_buffers_recv;

    int nb_processes_involved;
    bool current_is_involved;
    partsize_t particles_chunk_per_process;
    partsize_t particles_chunk_current_size;
    partsize_t particles_chunk_current_offset;

protected:
    MPI_Comm& getComWriter(){
        return mpi_com_writer;
    }

    partsize_t getTotalNbParticles() const {
        return total_nb_particles;
    }

    int getNbRhs() const {
        return nb_rhs;
    }

    int getMyRank(){
        return this->my_rank;
    }

    bool isInvolved() const{
        return current_is_involved;
    }

public:
    abstract_particles_output(MPI_Comm in_mpi_com, const partsize_t inTotalNbParticles, const int in_nb_rhs) throw()
            : mpi_com(in_mpi_com), my_rank(-1), nb_processes(-1),
                total_nb_particles(inTotalNbParticles), nb_rhs(in_nb_rhs),
                buffer_particles_rhs_send(in_nb_rhs), size_buffers_send(-1),
                buffer_particles_rhs_recv(in_nb_rhs), size_buffers_recv(-1),
                nb_processes_involved(0), current_is_involved(true), particles_chunk_per_process(0),
                particles_chunk_current_size(0), particles_chunk_current_offset(0) {

        AssertMpi(MPI_Comm_rank(mpi_com, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_com, &nb_processes));

        const size_t MinBytesPerProcess = env_utils::GetValue<size_t>("BFPS_PO_MIN_BYTES", 32 * 1024 * 1024); // Default 32MB
        const size_t ChunkBytes = env_utils::GetValue<size_t>("BFPS_PO_CHUNK_BYTES", 8 * 1024 * 1024); // Default 8MB
        const int MaxProcessesInvolved = std::min(nb_processes, env_utils::GetValue<int>("BFPS_PO_MAX_PROCESSES", 128));

        // We split the processes using positions size only
        const size_t totalBytesForPositions = total_nb_particles*size_particle_positions*sizeof(real_number);


        if(MinBytesPerProcess*MaxProcessesInvolved < totalBytesForPositions){
            size_t extraChunkBytes = 1;
            while((MinBytesPerProcess+extraChunkBytes*ChunkBytes)*MaxProcessesInvolved < totalBytesForPositions){
                extraChunkBytes += 1;
            }
            const size_t bytesPerProcess = (MinBytesPerProcess+extraChunkBytes*ChunkBytes);
            particles_chunk_per_process = partsize_t((bytesPerProcess+sizeof(real_number)*size_particle_positions-1)/(sizeof(real_number)*size_particle_positions));
            nb_processes_involved = int((total_nb_particles+particles_chunk_per_process-1)/particles_chunk_per_process);
        }
        // else limit based on minBytesPerProcess
        else{
            nb_processes_involved = std::max(1,std::min(MaxProcessesInvolved,int((totalBytesForPositions+MinBytesPerProcess-1)/MinBytesPerProcess)));
            particles_chunk_per_process = partsize_t((MinBytesPerProcess+sizeof(real_number)*size_particle_positions-1)/(sizeof(real_number)*size_particle_positions));
        }

        // Print out
        if(my_rank == 0){
            DEBUG_MSG("[INFO] Limit of processes involved in the particles ouput = %d (BFPS_PO_MAX_PROCESSES)\n", MaxProcessesInvolved);
            DEBUG_MSG("[INFO] Minimum bytes per process to write = %llu (BFPS_PO_MIN_BYTES) for a complete output of = %llu for positions\n", MinBytesPerProcess, totalBytesForPositions);
            DEBUG_MSG("[INFO] Consequently, there are %d processes that actually write data (%d particles per process)\n", nb_processes_involved, particles_chunk_per_process);
        }

        if(my_rank < nb_processes_involved){
            current_is_involved = true;
            particles_chunk_current_offset = my_rank*particles_chunk_per_process;
            assert(particles_chunk_current_offset < total_nb_particles);
            particles_chunk_current_size = std::min(particles_chunk_per_process, total_nb_particles-particles_chunk_current_offset);
            assert(particles_chunk_current_offset + particles_chunk_current_size <= total_nb_particles);
            assert(my_rank != nb_processes_involved-1 || particles_chunk_current_offset + particles_chunk_current_size == total_nb_particles);
        }
        else{
            current_is_involved = false;
            particles_chunk_current_size = 0;
            particles_chunk_current_offset = total_nb_particles;
        }

        AssertMpi( MPI_Comm_split(mpi_com,
                       (current_is_involved ? 1 : MPI_UNDEFINED),
                       my_rank, &mpi_com_writer) );
    }

    virtual ~abstract_particles_output(){
        if(current_is_involved){
            AssertMpi( MPI_Comm_free(&mpi_com_writer) );
        }
    }

    void releaseMemory(){
        buffer_indexes_send.release();
        buffer_particles_positions_send.release();
        size_buffers_send = -1;
        buffer_indexes_recv.release();
        buffer_particles_positions_recv.release();
        size_buffers_recv = -1;
        for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
            buffer_particles_rhs_send[idx_rhs].release();
            buffer_particles_rhs_recv[idx_rhs].release();
        }
    }

    void save(
            const real_number input_particles_positions[],
            const std::unique_ptr<real_number[]> input_particles_rhs[],
            const partsize_t index_particles[],
            const partsize_t nb_particles,
            const int idx_time_step){
        TIMEZONE("abstract_particles_output::save");
        assert(total_nb_particles != -1);

        {
            TIMEZONE("sort-to-distribute");

            if(size_buffers_send < nb_particles && nb_particles){
                buffer_indexes_send.reset(new std::pair<partsize_t,partsize_t>[nb_particles]);
                buffer_particles_positions_send.reset(new real_number[nb_particles*size_particle_positions]);
                for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                    buffer_particles_rhs_send[idx_rhs].reset(new real_number[nb_particles*size_particle_rhs]);
                }
                size_buffers_send = nb_particles;
            }

            for(partsize_t idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                buffer_indexes_send[idx_part].first = idx_part;
                buffer_indexes_send[idx_part].second = index_particles[idx_part];
            }

            std::sort(&buffer_indexes_send[0], &buffer_indexes_send[nb_particles], [](const std::pair<partsize_t,partsize_t>& p1,
                                                                                      const std::pair<partsize_t,partsize_t>& p2){
                return p1.second < p2.second;
            });

            for(partsize_t idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
                const partsize_t src_idx = buffer_indexes_send[idx_part].first;
                const partsize_t dst_idx = idx_part;

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

        partsize_t* buffer_indexes_send_tmp = reinterpret_cast<partsize_t*>(buffer_indexes_send.get());// trick re-use buffer_indexes_send memory
        std::vector<partsize_t> nb_particles_to_send(nb_processes, 0);
        for(partsize_t idx_part = 0 ; idx_part < nb_particles ; ++idx_part){
            const int dest_proc = int(buffer_indexes_send[idx_part].second/particles_chunk_per_process);
            assert(dest_proc < nb_processes_involved);
            nb_particles_to_send[dest_proc] += 1;
            buffer_indexes_send_tmp[idx_part] = buffer_indexes_send[idx_part].second;
        }

        alltoall_exchanger exchanger(mpi_com, std::move(nb_particles_to_send));
        // nb_particles_to_send is invalid after here

        const int nb_to_receive = exchanger.getTotalToRecv();
        assert(nb_to_receive == particles_chunk_current_size);

        if(size_buffers_recv < nb_to_receive && nb_to_receive){
            buffer_indexes_recv.reset(new partsize_t[nb_to_receive]);
            buffer_particles_positions_recv.reset(new real_number[nb_to_receive*size_particle_positions]);
            for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                buffer_particles_rhs_recv[idx_rhs].reset(new real_number[nb_to_receive*size_particle_rhs]);
            }
            size_buffers_recv = nb_to_receive;
        }

        {
            TIMEZONE("exchange");
            // Could be done with multiple asynchronous coms
            exchanger.alltoallv<partsize_t>(buffer_indexes_send_tmp, buffer_indexes_recv.get());
            exchanger.alltoallv<real_number>(buffer_particles_positions_send.get(), buffer_particles_positions_recv.get(), size_particle_positions);
            for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                exchanger.alltoallv<real_number>(buffer_particles_rhs_send[idx_rhs].get(), buffer_particles_rhs_recv[idx_rhs].get(), size_particle_rhs);
            }
        }

        // Stop here if not involved
        if(current_is_involved == false){
            assert(nb_to_receive == 0);
            return;
        }

        if(size_buffers_send < nb_to_receive && nb_to_receive){
            buffer_indexes_send.reset(new std::pair<partsize_t,partsize_t>[nb_to_receive]);
            buffer_particles_positions_send.reset(new real_number[nb_to_receive*size_particle_positions]);
            for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                buffer_particles_rhs_send[idx_rhs].reset(new real_number[nb_to_receive*size_particle_rhs]);
            }
            size_buffers_send = nb_to_receive;
        }

        {
            TIMEZONE("copy-local-order");
            for(partsize_t idx_part = 0 ; idx_part < nb_to_receive ; ++idx_part){
                const partsize_t src_idx = idx_part;
                const partsize_t dst_idx = buffer_indexes_recv[idx_part]-particles_chunk_current_offset;
                assert(0 <= dst_idx);
                assert(dst_idx < particles_chunk_current_size);

                for(int idx_val = 0 ; idx_val < size_particle_positions ; ++idx_val){
                    buffer_particles_positions_send[dst_idx*size_particle_positions + idx_val]
                            = buffer_particles_positions_recv[src_idx*size_particle_positions + idx_val];
                }
                for(int idx_rhs = 0 ; idx_rhs < nb_rhs ; ++idx_rhs){
                    for(int idx_val = 0 ; idx_val < int(size_particle_rhs) ; ++idx_val){
                        buffer_particles_rhs_send[idx_rhs][dst_idx*size_particle_rhs + idx_val]
                                = buffer_particles_rhs_recv[idx_rhs][src_idx*size_particle_rhs + idx_val];
                    }
                }
            }
        }

        write(idx_time_step, buffer_particles_positions_send.get(), buffer_particles_rhs_send.data(),
              nb_to_receive, particles_chunk_current_offset);
    }

    virtual void write(const int idx_time_step, const real_number* positions, const std::unique_ptr<real_number[]>* rhs,
                       const partsize_t nb_particles, const partsize_t particles_idx_offset) = 0;
};

#endif
