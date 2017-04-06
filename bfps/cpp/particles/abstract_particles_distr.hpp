#ifndef ABSTRACT_PARTICLES_DISTR_HPP
#define ABSTRACT_PARTICLES_DISTR_HPP

#include <mpi.h>

#include <vector>
#include <memory>
#include <cassert>

#include <type_traits>

#include "scope_timer.hpp"
#include "particles_utils.hpp"


template <class real_number, int size_particle_positions, int size_particle_rhs, int size_particle_index>
class abstract_particles_distr {
protected:
    static const int MaxNbRhs = 100;

    // Used withing the loop, allocate only once
    enum MpiTag{
        TAG_LOW_UP_NB_PARTICLES,
        TAG_UP_LOW_NB_PARTICLES,
        TAG_LOW_UP_PARTICLES,
        TAG_UP_LOW_PARTICLES,
        TAG_LOW_UP_RESULTS,
        TAG_UP_LOW_RESULTS,

        TAG_LOW_UP_MOVED_NB_PARTICLES,
        TAG_UP_LOW_MOVED_NB_PARTICLES,
        TAG_LOW_UP_MOVED_PARTICLES,
        TAG_UP_LOW_MOVED_PARTICLES,

        TAG_LOW_UP_MOVED_PARTICLES_INDEXES,
        TAG_UP_LOW_MOVED_PARTICLES_INDEXES,

        TAG_LOW_UP_MOVED_PARTICLES_RHS,
        TAG_LOW_UP_MOVED_PARTICLES_RHS_MAX = TAG_LOW_UP_MOVED_PARTICLES_RHS+MaxNbRhs,

        TAG_UP_LOW_MOVED_PARTICLES_RHS = TAG_LOW_UP_MOVED_PARTICLES_RHS_MAX,
        TAG_UP_LOW_MOVED_PARTICLES_RHS_MAX = TAG_UP_LOW_MOVED_PARTICLES_RHS+MaxNbRhs,
    };

    struct NeighborDescriptor{
        int nbPartitionsToSend;
        int nbPartitionsToRecv;
        int nbParticlesToSend;
        int nbParticlesToRecv;
        int destProc;
        int rankDiff;
        bool isLower;
        int idxLowerUpper;

        std::unique_ptr<real_number[]> toRecvAndMerge;
        std::unique_ptr<real_number[]> toCompute;
        std::unique_ptr<real_number[]> results;
    };

    enum Action{
        NOTHING_TODO,
        RECV_PARTICLES,
        COMPUTE_PARTICLES,
        RELEASE_BUFFER_PARTICLES,
        MERGE_PARTICLES,

        RECV_MOVE_NB_LOW,
        RECV_MOVE_NB_UP,
        RECV_MOVE_LOW,
        RECV_MOVE_UP
    };

    MPI_Comm current_com;

    int my_rank;
    int nb_processes;

    const std::pair<int,int> current_partition_interval;
    const int current_partition_size;

    std::unique_ptr<int[]> partition_interval_size_per_proc;
    std::unique_ptr<int[]> partition_interval_offset_per_proc;

    std::unique_ptr<int[]> current_offset_particles_for_partition;

    std::vector<std::pair<Action,int>> whatNext;
    std::vector<MPI_Request> mpiRequests;
    std::vector<NeighborDescriptor> neigDescriptors;

public:
    ////////////////////////////////////////////////////////////////////////////

    abstract_particles_distr(MPI_Comm in_current_com,
                             const std::pair<int,int>& in_current_partitions)
        : current_com(in_current_com),
            my_rank(-1), nb_processes(-1),
            current_partition_interval(in_current_partitions),
            current_partition_size(current_partition_interval.second-current_partition_interval.first){

        AssertMpi(MPI_Comm_rank(current_com, &my_rank));
        AssertMpi(MPI_Comm_size(current_com, &nb_processes));

        assert(current_partition_size >= 1);

        partition_interval_size_per_proc.reset(new int[nb_processes]);
        AssertMpi( MPI_Allgather( const_cast<int*>(&current_partition_size), 1, MPI_INT,
                                  partition_interval_size_per_proc.get(), 1, MPI_INT,
                                  current_com) );
        assert(partition_interval_size_per_proc[my_rank] == current_partition_size);

        partition_interval_offset_per_proc.reset(new int[nb_processes+1]);
        partition_interval_offset_per_proc[0] = 0;
        for(int idxProc = 0 ; idxProc < nb_processes ; ++idxProc){
            partition_interval_offset_per_proc[idxProc+1] = partition_interval_offset_per_proc[idxProc] + partition_interval_size_per_proc[idxProc];
        }

        current_offset_particles_for_partition.reset(new int[current_partition_size+1]);
    }

    virtual ~abstract_particles_distr(){}

    ////////////////////////////////////////////////////////////////////////////

    void compute_distr(const int current_my_nb_particles_per_partition[],
                       const real_number particles_positions[],
                       real_number particles_current_rhs[],
                       const int interpolation_size){
        TIMEZONE("compute_distr");

        current_offset_particles_for_partition[0] = 0;
        int myTotalNbParticles = 0;
        for(int idxPartition = 0 ; idxPartition < current_partition_size ; ++idxPartition){
            myTotalNbParticles += current_my_nb_particles_per_partition[idxPartition];
            current_offset_particles_for_partition[idxPartition+1] = current_offset_particles_for_partition[idxPartition] + current_my_nb_particles_per_partition[idxPartition];
        }

        //////////////////////////////////////////////////////////////////////
        /// Exchange the number of particles in each partition
        /// Could involve only here but I do not think it will be a problem
        //////////////////////////////////////////////////////////////////////


        assert(whatNext.size() == 0);
        assert(mpiRequests.size() == 0);

        neigDescriptors.clear();

        int nbProcToRecvLower;
        {
            int nextDestProc = my_rank;
            for(int idxLower = 1 ; idxLower <= interpolation_size ; idxLower += partition_interval_size_per_proc[nextDestProc]){
                nextDestProc = (nextDestProc-1+nb_processes)%nb_processes;
                const int destProc = nextDestProc;
                const int lowerRankDiff = (nextDestProc < my_rank ? my_rank - nextDestProc : nb_processes-nextDestProc+my_rank);

                const int nbPartitionsToSend = std::min(current_partition_size, interpolation_size-(idxLower-1));
                const int nbParticlesToSend = current_offset_particles_for_partition[nbPartitionsToSend] - current_offset_particles_for_partition[0];

                const int nbPartitionsToRecv = std::min(partition_interval_size_per_proc[destProc], (interpolation_size+1)-(idxLower-1));
                const int nbParticlesToRecv = -1;

                NeighborDescriptor descriptor;
                descriptor.destProc = destProc;
                descriptor.rankDiff = lowerRankDiff;
                descriptor.nbPartitionsToSend = nbPartitionsToSend;
                descriptor.nbParticlesToSend = nbParticlesToSend;
                descriptor.nbPartitionsToRecv = nbPartitionsToRecv;
                descriptor.nbParticlesToRecv = nbParticlesToRecv;
                descriptor.isLower = true;
                descriptor.idxLowerUpper = idxLower;

                neigDescriptors.emplace_back(std::move(descriptor));
            }
            nbProcToRecvLower = neigDescriptors.size();

            nextDestProc = my_rank;
            for(int idxUpper = 1 ; idxUpper <= interpolation_size ; idxUpper += partition_interval_size_per_proc[nextDestProc]){
                nextDestProc = (nextDestProc+1+nb_processes)%nb_processes;
                const int destProc = nextDestProc;
                const int upperRankDiff = (nextDestProc > my_rank ? nextDestProc - my_rank: nb_processes-my_rank+nextDestProc);

                const int nbPartitionsToSend = std::min(current_partition_size, (interpolation_size+1)-(idxUpper-1));
                const int nbParticlesToSend = current_offset_particles_for_partition[current_partition_size] - current_offset_particles_for_partition[current_partition_size-nbPartitionsToSend];

                const int nbPartitionsToRecv = std::min(partition_interval_size_per_proc[destProc], interpolation_size-(idxUpper-1));
                const int nbParticlesToRecv = -1;

                NeighborDescriptor descriptor;
                descriptor.destProc = destProc;
                descriptor.rankDiff = upperRankDiff;
                descriptor.nbPartitionsToSend = nbPartitionsToSend;
                descriptor.nbParticlesToSend = nbParticlesToSend;
                descriptor.nbPartitionsToRecv = nbPartitionsToRecv;
                descriptor.nbParticlesToRecv = nbParticlesToRecv;
                descriptor.isLower = false;
                descriptor.idxLowerUpper = idxUpper;

                neigDescriptors.emplace_back(std::move(descriptor));
            }
        }
        const int nbProcToRecvUpper = neigDescriptors.size()-nbProcToRecvLower;
        const int nbProcToRecv = nbProcToRecvUpper + nbProcToRecvLower;
        assert(neigDescriptors.size() == nbProcToRecv);
        DEBUG_MSG("[%d] nbProcToRecvUpper %d\n", my_rank, nbProcToRecvUpper);
        DEBUG_MSG("[%d] nbProcToRecvLower %d\n", my_rank, nbProcToRecvLower);
        DEBUG_MSG("[%d] nbProcToRecv %d\n", my_rank, nbProcToRecv);

        for(int idxDescr = 0 ; idxDescr < neigDescriptors.size() ; ++idxDescr){
            NeighborDescriptor& descriptor = neigDescriptors[idxDescr];

            if(descriptor.isLower){
                DEBUG_MSG("[%d] Send idxLower %d  -- nbPartitionsToSend %d -- nbParticlesToSend %d\n",
                       my_rank, descriptor.idxLowerUpper, descriptor.nbPartitionsToSend, descriptor.nbParticlesToSend);
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(const_cast<int*>(&descriptor.nbParticlesToSend), 1, MPI_INT, descriptor.destProc, TAG_LOW_UP_NB_PARTICLES,
                          current_com, &mpiRequests.back()));

                if(descriptor.nbParticlesToSend){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Isend(const_cast<real_number*>(&particles_positions[0]), descriptor.nbParticlesToSend*size_particle_positions, particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_LOW_UP_PARTICLES,
                              current_com, &mpiRequests.back()));

                    assert(descriptor.toRecvAndMerge == nullptr);
                    descriptor.toRecvAndMerge.reset(new real_number[descriptor.nbParticlesToSend*size_particle_rhs]);
                    whatNext.emplace_back(std::pair<Action,int>{MERGE_PARTICLES, idxDescr});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Irecv(descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend*size_particle_rhs, particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_UP_LOW_RESULTS,
                              current_com, &mpiRequests.back()));
                }

                whatNext.emplace_back(std::pair<Action,int>{RECV_PARTICLES, idxDescr});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Irecv(&descriptor.nbParticlesToRecv,
                          1, MPI_INT, descriptor.destProc, TAG_UP_LOW_NB_PARTICLES,
                          current_com, &mpiRequests.back()));
            }
            else{
                DEBUG_MSG("[%d] Send idxUpper %d  -- nbPartitionsToSend %d -- nbParticlesToSend %d\n",
                       my_rank, descriptor.idxLowerUpper, descriptor.nbPartitionsToSend, descriptor.nbParticlesToSend);
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(const_cast<int*>(&descriptor.nbParticlesToSend), 1, MPI_INT, descriptor.destProc, TAG_UP_LOW_NB_PARTICLES,
                          current_com, &mpiRequests.back()));

                if(descriptor.nbParticlesToSend){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Isend(const_cast<real_number*>(&particles_positions[current_offset_particles_for_partition[current_partition_size-descriptor.nbPartitionsToSend]]), descriptor.nbParticlesToSend*size_particle_positions, particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_UP_LOW_PARTICLES,
                              current_com, &mpiRequests.back()));

                    assert(descriptor.toRecvAndMerge == nullptr);
                    descriptor.toRecvAndMerge.reset(new real_number[descriptor.nbParticlesToSend*size_particle_rhs]);
                    whatNext.emplace_back(std::pair<Action,int>{MERGE_PARTICLES, idxDescr});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Irecv(descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend*size_particle_rhs, particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_LOW_UP_RESULTS,
                              current_com, &mpiRequests.back()));
                }

                whatNext.emplace_back(std::pair<Action,int>{RECV_PARTICLES, idxDescr});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Irecv(&descriptor.nbParticlesToRecv,
                          1, MPI_INT, descriptor.destProc, TAG_LOW_UP_NB_PARTICLES,
                          current_com, &mpiRequests.back()));
            }
        }

        while(mpiRequests.size()){
            assert(mpiRequests.size() == whatNext.size());

            int idxDone = mpiRequests.size();
            {
                TIMEZONE("wait");
                AssertMpi(MPI_Waitany(mpiRequests.size(), mpiRequests.data(), &idxDone, MPI_STATUSES_IGNORE));
            }
            const std::pair<Action, int> releasedAction = whatNext[idxDone];
            std::swap(mpiRequests[idxDone], mpiRequests[mpiRequests.size()-1]);
            std::swap(whatNext[idxDone], whatNext[mpiRequests.size()-1]);
            mpiRequests.pop_back();
            whatNext.pop_back();

            //////////////////////////////////////////////////////////////////////
            /// Data to exchange particles
            //////////////////////////////////////////////////////////////////////
            if(releasedAction.first == RECV_PARTICLES){
                DEBUG_MSG("[%d] RECV_PARTICLES\n", my_rank);
                NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];

                if(descriptor.isLower){
                    const int idxLower = descriptor.idxLowerUpper;
                    const int destProc = descriptor.destProc;
                    const int nbPartitionsToRecv = descriptor.nbPartitionsToRecv;
                    const int NbParticlesToReceive = descriptor.nbParticlesToRecv;
                    assert(NbParticlesToReceive != -1);
                    assert(descriptor.toCompute == nullptr);
                    DEBUG_MSG("[%d] Recv idxLower %d  -- nbPartitionsToRecv %d -- NbParticlesToReceive %d\n",
                           my_rank, idxLower, nbPartitionsToRecv, NbParticlesToReceive);
                    if(NbParticlesToReceive){
                        descriptor.toCompute.reset(new real_number[NbParticlesToReceive*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{COMPUTE_PARTICLES, releasedAction.second});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(descriptor.toCompute.get(), NbParticlesToReceive*size_particle_positions, particles_utils::GetMpiType(real_number()), destProc, TAG_UP_LOW_PARTICLES,
                                  current_com, &mpiRequests.back()));
                    }
                }
                else{
                    const int idxUpper = descriptor.idxLowerUpper;
                    const int destProc = descriptor.destProc;
                    const int nbPartitionsToRecv = descriptor.nbPartitionsToRecv;
                    const int NbParticlesToReceive = descriptor.nbParticlesToRecv;
                    assert(NbParticlesToReceive != -1);
                    assert(descriptor.toCompute == nullptr);
                    DEBUG_MSG("[%d] Recv idxUpper %d  -- nbPartitionsToRecv %d -- NbParticlesToReceive %d\n",
                           my_rank, idxUpper, nbPartitionsToRecv, NbParticlesToReceive);
                    if(NbParticlesToReceive){
                        descriptor.toCompute.reset(new real_number[NbParticlesToReceive*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{COMPUTE_PARTICLES, releasedAction.second});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(descriptor.toCompute.get(), NbParticlesToReceive*size_particle_positions, particles_utils::GetMpiType(real_number()), destProc, TAG_LOW_UP_PARTICLES,
                                  current_com, &mpiRequests.back()));
                    }
                }
            }

            //////////////////////////////////////////////////////////////////////
            /// Computation
            //////////////////////////////////////////////////////////////////////
            if(releasedAction.first == COMPUTE_PARTICLES){
                DEBUG_MSG("[%d] COMPUTE_PARTICLES\n", my_rank);
                NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];
                const int NbParticlesToReceive = descriptor.nbParticlesToRecv;

                assert(descriptor.toCompute != nullptr);
                descriptor.results.reset(new real_number[NbParticlesToReceive*size_particle_rhs]);
                init_result_array(descriptor.results.get(), NbParticlesToReceive);
                apply_computation(descriptor.toCompute.get(), descriptor.results.get(), NbParticlesToReceive);

                const int destProc = descriptor.destProc;
                whatNext.emplace_back(std::pair<Action,int>{RELEASE_BUFFER_PARTICLES, releasedAction.second});
                mpiRequests.emplace_back();
                const int tag = descriptor.isLower? TAG_LOW_UP_RESULTS : TAG_UP_LOW_RESULTS;
                AssertMpi(MPI_Isend(descriptor.results.get(), NbParticlesToReceive*size_particle_rhs, particles_utils::GetMpiType(real_number()), destProc, tag,
                          current_com, &mpiRequests.back()));
            }
            //////////////////////////////////////////////////////////////////////
            /// Computation
            //////////////////////////////////////////////////////////////////////
            if(releasedAction.first == RELEASE_BUFFER_PARTICLES){
                DEBUG_MSG("[%d] RELEASE_BUFFER_PARTICLES\n", my_rank);
                NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];
                assert(descriptor.toCompute != nullptr);
                descriptor.toCompute.release();
            }
            //////////////////////////////////////////////////////////////////////
            /// Merge
            //////////////////////////////////////////////////////////////////////
            if(releasedAction.first == MERGE_PARTICLES){
                DEBUG_MSG("[%d] MERGE_PARTICLES\n", my_rank);
                NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];

                if(descriptor.isLower){
                    DEBUG_MSG("[%d] low buffer received\n", my_rank);
                    TIMEZONE("reduce");
                    assert(descriptor.toRecvAndMerge != nullptr);
                    reduce_particles(&particles_positions[0], &particles_current_rhs[0], descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                    descriptor.toRecvAndMerge.release();
                }
                else {
                    DEBUG_MSG("[%d] up buffer received\n", my_rank);
                    TIMEZONE("reduce");
                    assert(descriptor.toRecvAndMerge != nullptr);
                    reduce_particles(&particles_positions[current_offset_particles_for_partition[current_partition_size]-descriptor.nbParticlesToSend],
                                     &particles_current_rhs[current_offset_particles_for_partition[current_partition_size]-descriptor.nbParticlesToSend],
                                     descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                    descriptor.toRecvAndMerge.release();
                }
            }
        }


        {
            // TODO compute only border sections and do the reste in parallel
            TIMEZONE("compute");
            // Compute my particles
            if(myTotalNbParticles){
                TIMEZONE("my_compute");
                apply_computation(particles_positions, particles_current_rhs, myTotalNbParticles);
            }
        }

        assert(mpiRequests.size() == 0);
    }

    ////////////////////////////////////////////////////////////////////////////

    virtual void init_result_array(real_number particles_current_rhs[],
                                   const int nb_particles) = 0;
    virtual void apply_computation(const real_number particles_positions[],
                                   real_number particles_current_rhs[],
                                   const int nb_particles) const = 0;
    virtual void reduce_particles(const real_number particles_positions[],
                                  real_number particles_current_rhs[],
                                  const real_number extra_particles_current_rhs[],
                                  const int nb_particles) const = 0;

    ////////////////////////////////////////////////////////////////////////////

    void redistribute(int current_my_nb_particles_per_partition[],
                      int* nb_particles,
                      std::unique_ptr<real_number[]>* inout_positions_particles,
                      std::unique_ptr<real_number[]> inout_rhs_particles[], const int in_nb_rhs,
                      std::unique_ptr<int[]>* inout_index_particles,
                      const real_number mySpatialLowLimit,
                      const real_number mySpatialUpLimit,
                      const real_number spatialPartitionWidth,
                      const int myTotalNbParticlesAllocated=-1){
        TIMEZONE("redistribute");
        current_offset_particles_for_partition[0] = 0;
        int myTotalNbParticles = 0;
        for(int idxPartition = 0 ; idxPartition < current_partition_size ; ++idxPartition){
            myTotalNbParticles += current_my_nb_particles_per_partition[idxPartition];
            current_offset_particles_for_partition[idxPartition+1] = current_offset_particles_for_partition[idxPartition] + current_my_nb_particles_per_partition[idxPartition];
        }
        assert((*nb_particles) == myTotalNbParticles);

        // Find particles outside my interval
        const int nbOutLower = particles_utils::partition_extra<size_particle_positions>(&(*inout_positions_particles)[0], current_my_nb_particles_per_partition[0],
                    [&](const real_number val[]){
            const bool isLower = val[IDX_Z] < mySpatialLowLimit;
            return isLower;
        },
                    [&](const int idx1, const int idx2){
            for(int idx_val = 0 ; idx_val < size_particle_index ; ++idx_val){
                std::swap((*inout_index_particles)[idx1], (*inout_index_particles)[idx2]);
            }

            for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    std::swap(inout_rhs_particles[idx_rhs][idx1*size_particle_rhs + idx_val],
                              inout_rhs_particles[idx_rhs][idx2*size_particle_rhs + idx_val]);
                }
            }
        });
        DEBUG_MSG("[%d] nbOutLower %d\n", my_rank, nbOutLower);

        const int offesetOutLow = (current_partition_size==1? nbOutLower : 0);

        const int nbOutUpper = current_my_nb_particles_per_partition[current_partition_size-1] - offesetOutLow - (particles_utils::partition_extra<size_particle_positions>(
                    &(*inout_positions_particles)[(current_offset_particles_for_partition[current_partition_size-1]+offesetOutLow)*size_particle_positions],
                    myTotalNbParticles - (current_offset_particles_for_partition[current_partition_size-1]+offesetOutLow),
                    [&](const real_number val[]){
            const bool isUpper = mySpatialUpLimit <= val[IDX_Z];
            return !isUpper;
        },
                    [&](const int idx1, const int idx2){
            for(int idx_val = 0 ; idx_val < size_particle_index ; ++idx_val){
                std::swap((*inout_index_particles)[idx1], (*inout_index_particles)[idx2]);
            }

            for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    std::swap(inout_rhs_particles[idx_rhs][idx1*size_particle_rhs + idx_val],
                              inout_rhs_particles[idx_rhs][idx2*size_particle_rhs + idx_val]);
                }
            }
        }));
        DEBUG_MSG("[%d] nbOutUpper %d\n", my_rank, nbOutUpper);

        // Exchange number
        int eventsBeforeWaitall = 0;
        int nbNewFromLow = 0;
        int nbNewFromUp = 0;
        std::unique_ptr<real_number[]> newParticlesLow;
        std::unique_ptr<real_number[]> newParticlesUp;
        std::unique_ptr<int[]> newParticlesLowIndexes;
        std::unique_ptr<int[]> newParticlesUpIndexes;
        std::vector<std::unique_ptr<real_number[]>> newParticlesLowRhs(in_nb_rhs);
        std::vector<std::unique_ptr<real_number[]>> newParticlesUpRhs(in_nb_rhs);

        {
            assert(whatNext.size() == 0);
            assert(mpiRequests.size() == 0);

            whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_NB_LOW, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Irecv(&nbNewFromLow, 1, MPI_INT, (my_rank-1+nb_processes)%nb_processes, TAG_UP_LOW_MOVED_NB_PARTICLES,
                      MPI_COMM_WORLD, &mpiRequests.back()));
            eventsBeforeWaitall += 1;

            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Isend(const_cast<int*>(&nbOutLower), 1, MPI_INT, (my_rank-1+nb_processes)%nb_processes, TAG_LOW_UP_MOVED_NB_PARTICLES,
                      MPI_COMM_WORLD, &mpiRequests.back()));

            if(nbOutLower){
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(&(*inout_positions_particles)[0], nbOutLower*size_particle_positions, particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES,
                          MPI_COMM_WORLD, &mpiRequests.back()));
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(&(*inout_index_particles)[0], nbOutLower, MPI_INT, (my_rank-1+nb_processes)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES_INDEXES,
                          MPI_COMM_WORLD, &mpiRequests.back()));

                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Isend(&inout_rhs_particles[idx_rhs][0], nbOutLower*size_particle_rhs, particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES_RHS+idx_rhs,
                              MPI_COMM_WORLD, &mpiRequests.back()));
                }
            }

            whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_NB_UP, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Irecv(&nbNewFromUp, 1, MPI_INT, (my_rank+1)%nb_processes, TAG_LOW_UP_MOVED_NB_PARTICLES,
                      MPI_COMM_WORLD, &mpiRequests.back()));
            eventsBeforeWaitall += 1;

            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Isend(const_cast<int*>(&nbOutUpper), 1, MPI_INT, (my_rank+1)%nb_processes, TAG_UP_LOW_MOVED_NB_PARTICLES,
                      MPI_COMM_WORLD, &mpiRequests.back()));

            if(nbOutUpper){
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(&(*inout_positions_particles)[(myTotalNbParticles-nbOutUpper)*size_particle_positions], nbOutUpper*size_particle_positions, particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES,
                          MPI_COMM_WORLD, &mpiRequests.back()));
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(&(*inout_index_particles)[(myTotalNbParticles-nbOutUpper)], nbOutUpper, MPI_INT, (my_rank+1)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES_INDEXES,
                          MPI_COMM_WORLD, &mpiRequests.back()));


                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Isend(&inout_rhs_particles[idx_rhs][(myTotalNbParticles-nbOutUpper)*size_particle_rhs], nbOutUpper*size_particle_rhs, particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES_RHS+idx_rhs,
                              MPI_COMM_WORLD, &mpiRequests.back()));
                }
            }

            while(mpiRequests.size() && eventsBeforeWaitall){
                DEBUG_MSG("eventsBeforeWaitall %d\n", eventsBeforeWaitall);

                int idxDone = mpiRequests.size();
                {
                    TIMEZONE("waitany_move");
                    AssertMpi(MPI_Waitany(mpiRequests.size(), mpiRequests.data(), &idxDone, MPI_STATUSES_IGNORE));
                    DEBUG_MSG("MPI_Waitany eventsBeforeWaitall %d\n", eventsBeforeWaitall);
                }
                const std::pair<Action, int> releasedAction = whatNext[idxDone];
                std::swap(mpiRequests[idxDone], mpiRequests[mpiRequests.size()-1]);
                std::swap(whatNext[idxDone], whatNext[mpiRequests.size()-1]);
                mpiRequests.pop_back();
                whatNext.pop_back();

                if(releasedAction.first == RECV_MOVE_NB_LOW){
                    DEBUG_MSG("[%d] nbNewFromLow %d from %d\n", my_rank, nbNewFromLow, (my_rank-1+nb_processes)%nb_processes);

                    if(nbNewFromLow){
                        assert(newParticlesLow == nullptr);
                        newParticlesLow.reset(new real_number[nbNewFromLow*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_LOW, -1});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(&newParticlesLow[0], nbNewFromLow*size_particle_positions, particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        newParticlesLowIndexes.reset(new int[nbNewFromLow]);
                        whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(&newParticlesLowIndexes[0], nbNewFromLow, MPI_INT, (my_rank-1+nb_processes)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES_INDEXES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            newParticlesLowRhs[idx_rhs].reset(new real_number[nbNewFromLow*size_particle_rhs]);
                            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                            mpiRequests.emplace_back();
                            AssertMpi(MPI_Irecv(&newParticlesLowRhs[idx_rhs][0], nbNewFromLow*size_particle_rhs, particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes)%nb_processes, TAG_UP_LOW_MOVED_PARTICLES_RHS+idx_rhs,
                                      MPI_COMM_WORLD, &mpiRequests.back()));
                        }
                    }
                    eventsBeforeWaitall -= 1;
                }
                else if(releasedAction.first == RECV_MOVE_NB_UP){
                    DEBUG_MSG("[%d] nbNewFromUp %d from %d\n", my_rank, nbNewFromUp, (my_rank+1)%nb_processes);

                    if(nbNewFromUp){
                        assert(newParticlesUp == nullptr);
                        newParticlesUp.reset(new real_number[nbNewFromUp*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_UP, -1});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(&newParticlesUp[0], nbNewFromUp*size_particle_positions, particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        newParticlesUpIndexes.reset(new int[nbNewFromUp]);
                        whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                        mpiRequests.emplace_back();
                        AssertMpi(MPI_Irecv(&newParticlesUpIndexes[0], nbNewFromUp, MPI_INT, (my_rank+1)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES_INDEXES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            newParticlesUpRhs[idx_rhs].reset(new real_number[nbNewFromUp*size_particle_rhs]);
                            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                            mpiRequests.emplace_back();
                            AssertMpi(MPI_Irecv(&newParticlesUpRhs[idx_rhs][0], nbNewFromUp*size_particle_rhs, particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes, TAG_LOW_UP_MOVED_PARTICLES_RHS+idx_rhs,
                                      MPI_COMM_WORLD, &mpiRequests.back()));
                        }
                    }
                    eventsBeforeWaitall -= 1;
                }
            }
        }

        if(mpiRequests.size()){
            DEBUG_MSG("MPI_Waitall\n");
            // TODO Proceed when received
            TIMEZONE("waitall-move");
            AssertMpi(MPI_Waitall(mpiRequests.size(), mpiRequests.data(), MPI_STATUSES_IGNORE));
            mpiRequests.clear();
        }

        // Exchange particles
        {
            TIMEZONE("apply_pbc_z_new_particles");
            if(nbNewFromLow){
                assert(newParticlesLow.get() != nullptr);
                apply_pbc_z_new_particles(newParticlesLow.get(), nbNewFromLow);
            }
            if(nbNewFromUp){
                assert(newParticlesUp.get() != nullptr);
                apply_pbc_z_new_particles(newParticlesUp.get(), nbNewFromUp);
            }
        }

        // Realloc an merge
        if(nbNewFromLow + nbNewFromUp != 0){
            TIMEZONE("realloc_and_merge");
            const int nbOldParticlesInside = myTotalNbParticles - nbOutLower - nbOutUpper;
            const int myTotalNewNbParticles = nbOldParticlesInside + nbNewFromLow + nbNewFromUp;

            DEBUG_MSG("[%d] nbOldParticlesInside %d\n", my_rank, nbOldParticlesInside);
            DEBUG_MSG("[%d] myTotalNewNbParticles %d\n", my_rank, myTotalNewNbParticles);

            if(myTotalNewNbParticles > myTotalNbParticlesAllocated){
                DEBUG_MSG("[%d] reuse array\n", my_rank);
                std::unique_ptr<real_number[]> newArray(new real_number[myTotalNewNbParticles*size_particle_positions]);
                std::unique_ptr<int[]> newArrayIndexes(new int[myTotalNewNbParticles]);
                std::vector<std::unique_ptr<real_number[]>> newArrayRhs(in_nb_rhs);
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    newArrayRhs[idx_rhs].reset(new real_number[myTotalNewNbParticles*size_particle_rhs]);
                }

                if(nbNewFromLow){
                    memcpy(&newArray[0], &newParticlesLow[0], sizeof(real_number)*nbNewFromLow*size_particle_positions);
                    memcpy(&newArrayIndexes[0], &newParticlesLowIndexes[0], sizeof(int)*nbNewFromLow);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&newArrayRhs[idx_rhs][0], &newParticlesLowRhs[idx_rhs][0], sizeof(real_number)*nbNewFromLow*size_particle_rhs);
                    }
                }

                memcpy(&newArray[nbNewFromLow*size_particle_positions], &(*inout_positions_particles)[nbOutLower*size_particle_positions], sizeof(real_number)*nbOldParticlesInside*size_particle_positions);
                memcpy(&newArrayIndexes[nbNewFromLow], &(*inout_positions_particles)[nbOutLower], sizeof(int)*nbOldParticlesInside);
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    memcpy(&newArrayRhs[idx_rhs][nbNewFromLow*size_particle_rhs], &inout_positions_particles[idx_rhs][nbOutLower*size_particle_rhs], sizeof(real_number)*nbOldParticlesInside*size_particle_rhs);
                }

                if(nbNewFromUp){
                    memcpy(&newArray[(nbNewFromLow+nbOldParticlesInside)*size_particle_positions], &newParticlesUp[0], sizeof(real_number)*nbNewFromUp*size_particle_positions);
                    memcpy(&newArrayIndexes[(nbNewFromLow+nbOldParticlesInside)], &newParticlesUpIndexes[0], sizeof(int)*nbNewFromUp);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&newArrayRhs[idx_rhs][(nbNewFromLow+nbOldParticlesInside)*size_particle_rhs], &newParticlesUpRhs[idx_rhs][0], sizeof(real_number)*nbNewFromUp*size_particle_rhs);
                    }
                }

                (*inout_positions_particles) = std::move(newArray);
                (*inout_index_particles) = std::move(newArrayIndexes);
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    inout_rhs_particles[idx_rhs] = std::move(newArrayRhs[idx_rhs]);
                }

                // not needed myTotalNbParticlesAllocated = myTotalNewNbParticles;
            }
            else if(nbOutLower < nbNewFromLow){
                DEBUG_MSG("[%d] A array\n", my_rank);
                // Less low send thant received from low
                const int nbLowToMoveBack = nbNewFromLow-nbOutLower;
                // Copy received from low in two part
                if(nbNewFromLow){
                    memcpy(&(*inout_positions_particles)[0], &newParticlesLow[0], sizeof(real_number)*nbOutLower*size_particle_positions);
                    memcpy(&(*inout_index_particles)[0], &newParticlesLowIndexes[0], sizeof(int)*nbOutLower);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&inout_rhs_particles[idx_rhs][0], &newParticlesLowRhs[idx_rhs][0], sizeof(real_number)*nbOutLower*size_particle_rhs);
                    }
                }
                if(nbNewFromLow){
                    memcpy(&(*inout_positions_particles)[(nbOutLower+nbOldParticlesInside)*size_particle_positions], &newParticlesLow[nbOutLower*size_particle_positions], sizeof(real_number)*nbLowToMoveBack*size_particle_positions);
                    memcpy(&(*inout_index_particles)[(nbOutLower+nbOldParticlesInside)], &newParticlesLowIndexes[nbOutLower], sizeof(int)*nbLowToMoveBack);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&inout_rhs_particles[idx_rhs][(nbOutLower+nbOldParticlesInside)*size_particle_rhs], &newParticlesLowRhs[idx_rhs][nbOutLower*size_particle_rhs], sizeof(real_number)*nbLowToMoveBack*size_particle_rhs);
                    }
                }
                if(nbNewFromUp){
                    memcpy(&(*inout_positions_particles)[(nbNewFromLow+nbOldParticlesInside)*size_particle_positions], &newParticlesUp[0], sizeof(real_number)*nbNewFromUp*size_particle_positions);
                    memcpy(&(*inout_index_particles)[(nbNewFromLow+nbOldParticlesInside)], &newParticlesUpIndexes[0], sizeof(int)*nbNewFromUp);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&inout_rhs_particles[idx_rhs][(nbNewFromLow+nbOldParticlesInside)*size_particle_rhs], &newParticlesUpRhs[0], sizeof(real_number)*nbNewFromUp*size_particle_rhs);
                    }
                }
            }
            else{
                const int nbUpToMoveBegin = nbOutLower - nbNewFromLow;
                if(nbUpToMoveBegin <= nbNewFromUp){
                    DEBUG_MSG("[%d] B array\n", my_rank);
                    if(nbNewFromLow){
                        memcpy(&(*inout_positions_particles)[0], &newParticlesLow[0], sizeof(real_number)*nbNewFromLow*size_particle_positions);
                        memcpy(&(*inout_index_particles)[0], &newParticlesLowIndexes[0], sizeof(int)*nbNewFromLow);
                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            memcpy(&inout_rhs_particles[idx_rhs][0], &newParticlesLowRhs[idx_rhs][0], sizeof(real_number)*nbNewFromLow*size_particle_rhs);
                        }
                    }
                    if(nbNewFromUp){
                        memcpy(&(*inout_positions_particles)[nbNewFromLow*size_particle_positions], &newParticlesUp[0], sizeof(real_number)*nbUpToMoveBegin*size_particle_positions);
                        memcpy(&(*inout_index_particles)[nbNewFromLow], &newParticlesUpIndexes[0], sizeof(int)*nbUpToMoveBegin);
                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            memcpy(&inout_rhs_particles[idx_rhs][nbNewFromLow*size_particle_rhs], &newParticlesLowRhs[idx_rhs][0], sizeof(real_number)*nbUpToMoveBegin*size_particle_rhs);
                        }
                    }
                    if(nbNewFromUp){
                        memcpy(&(*inout_positions_particles)[(nbOutLower+nbOldParticlesInside)*size_particle_positions],
                                &newParticlesUp[nbUpToMoveBegin*size_particle_positions],
                                        sizeof(real_number)*(nbNewFromUp-nbUpToMoveBegin)*size_particle_positions);
                        memcpy(&(*inout_index_particles)[(nbOutLower+nbOldParticlesInside)], &newParticlesUpIndexes[nbUpToMoveBegin],
                                        sizeof(int)*(nbNewFromUp-nbUpToMoveBegin));
                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            memcpy(&inout_rhs_particles[idx_rhs][(nbOutLower+nbOldParticlesInside)*size_particle_rhs],
                                   &newParticlesUpRhs[idx_rhs][nbUpToMoveBegin*size_particle_rhs],
                                   sizeof(real_number)*(nbNewFromUp-nbUpToMoveBegin)*size_particle_rhs);
                        }
                    }
                }
                else{
                    DEBUG_MSG("[%d] C array\n", my_rank);
                    if(nbNewFromLow){
                        memcpy(&(*inout_positions_particles)[0], &newParticlesLow[0], sizeof(real_number)*nbNewFromLow*size_particle_positions);
                        memcpy(&(*inout_index_particles)[0], &newParticlesLowIndexes[0], sizeof(int)*nbNewFromLow);
                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            memcpy(&inout_rhs_particles[idx_rhs][0], &newParticlesLowRhs[idx_rhs][0], sizeof(real_number)*nbNewFromLow*size_particle_rhs);
                        }
                    }
                    if(nbNewFromUp){
                        memcpy(&(*inout_positions_particles)[0], &newParticlesUp[0], sizeof(real_number)*nbNewFromUp*size_particle_positions);
                        memcpy(&(*inout_index_particles)[0], &newParticlesUpIndexes[0], sizeof(int)*nbNewFromUp);
                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            memcpy(&inout_rhs_particles[idx_rhs][0], &newParticlesUpRhs[idx_rhs][0], sizeof(real_number)*nbNewFromUp*size_particle_rhs);
                        }
                    }
                    const int padding = nbOutLower - nbNewFromLow+nbNewFromUp;
                    memcpy(&(*inout_positions_particles)[(nbNewFromLow+nbNewFromUp)*size_particle_positions],
                            &(*inout_positions_particles)[(nbOutLower+nbOldParticlesInside-padding)*size_particle_positions],
                            sizeof(real_number)*padding*size_particle_positions);
                    memcpy(&(*inout_index_particles)[(nbNewFromLow+nbNewFromUp)],
                            &(*inout_index_particles)[(nbOutLower+nbOldParticlesInside-padding)],
                            sizeof(int)*padding);
                    for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                        memcpy(&inout_rhs_particles[idx_rhs][(nbNewFromLow+nbNewFromUp)*size_particle_rhs],
                                &inout_rhs_particles[idx_rhs][(nbOutLower+nbOldParticlesInside-padding)*size_particle_rhs],
                                sizeof(real_number)*padding*size_particle_rhs);
                    }
                }
            }
            myTotalNbParticles = myTotalNewNbParticles;
        }

        {
            TIMEZONE("apply_pbc_xy");
            apply_pbc_xy((*inout_positions_particles).get(), nbNewFromUp+nbNewFromLow);
        }

        // Partitions all particles
        {
            TIMEZONE("repartition");
            particles_utils::partition_extra_z<size_particle_positions>(&(*inout_positions_particles)[0],
                                             myTotalNbParticles,current_partition_size,
                                             current_my_nb_particles_per_partition, current_offset_particles_for_partition.get(),
                                             [&](const int idxPartition){
                return (idxPartition+1)*spatialPartitionWidth + mySpatialLowLimit;
            },
            [&](const int idx1, const int idx2){
                for(int idx_val = 0 ; idx_val < size_particle_index ; ++idx_val){
                    std::swap((*inout_index_particles)[idx1], (*inout_index_particles)[idx2]);
                }

                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                        std::swap(inout_rhs_particles[idx_rhs][idx1*size_particle_rhs + idx_val],
                                  inout_rhs_particles[idx_rhs][idx2*size_particle_rhs + idx_val]);
                    }
                }
            });

            {// TODO remove
                for(int idxPartition = 0 ; idxPartition < current_partition_size ; ++idxPartition){
                    assert(current_my_nb_particles_per_partition[idxPartition] ==
                           current_offset_particles_for_partition[idxPartition+1] - current_offset_particles_for_partition[idxPartition]);
                    const real_number limitPartition = (idxPartition+1)*spatialPartitionWidth + mySpatialLowLimit;
                    for(int idx = 0 ; idx < current_offset_particles_for_partition[idxPartition+1] ; ++idx){
                        assert((*inout_positions_particles)[idx*3+IDX_Z] < limitPartition);
                    }
                    for(int idx = current_offset_particles_for_partition[idxPartition+1] ; idx < myTotalNbParticles ; ++idx){
                        assert((*inout_positions_particles)[idx*3+IDX_Z] >= limitPartition);
                    }
                }
            }
        }
        (*nb_particles) = myTotalNbParticles;

        assert(mpiRequests.size() == 0);
    }

    virtual void apply_pbc_z_new_particles(real_number* newParticlesLow, const int nbNewFromLow) const = 0;
    virtual void apply_pbc_xy(real_number* inout_positions_particles, const int nbNew) const = 0;

    ////////////////////////////////////////////////////////////////////////////

    virtual void move_particles(real_number particles_positions[],
              const int nb_particles,
              const std::unique_ptr<real_number[]> particles_current_rhs[],
              const int nb_rhs, const real_number dt) const = 0;
};

#endif
