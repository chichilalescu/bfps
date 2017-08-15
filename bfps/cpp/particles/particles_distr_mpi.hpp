#ifndef PARTICLES_DISTR_MPI_HPP
#define PARTICLES_DISTR_MPI_HPP

#include <mpi.h>

#include <vector>
#include <memory>
#include <cassert>

#include <type_traits>
#include <omp.h>

#include "scope_timer.hpp"
#include "particles_utils.hpp"


template <class partsize_t, class real_number>
class particles_distr_mpi {
protected:
    static const int MaxNbRhs = 100;

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
        partsize_t nbParticlesToSend;
        partsize_t nbParticlesToRecv;
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
    int nb_processes_involved;

    const std::pair<int,int> current_partition_interval;
    const int current_partition_size;
    const std::array<size_t,3> field_grid_dim;

    std::unique_ptr<int[]> partition_interval_size_per_proc;
    std::unique_ptr<int[]> partition_interval_offset_per_proc;

    std::unique_ptr<partsize_t[]> current_offset_particles_for_partition;

    std::vector<std::pair<Action,int>> whatNext;
    std::vector<MPI_Request> mpiRequests;
    std::vector<NeighborDescriptor> neigDescriptors;

public:
    ////////////////////////////////////////////////////////////////////////////

    particles_distr_mpi(MPI_Comm in_current_com,
                             const std::pair<int,int>& in_current_partitions,
                             const std::array<size_t,3>& in_field_grid_dim)
        : current_com(in_current_com),
            my_rank(-1), nb_processes(-1),nb_processes_involved(-1),
            current_partition_interval(in_current_partitions),
            current_partition_size(current_partition_interval.second-current_partition_interval.first),
            field_grid_dim(in_field_grid_dim){

        AssertMpi(MPI_Comm_rank(current_com, &my_rank));
        AssertMpi(MPI_Comm_size(current_com, &nb_processes));

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

        current_offset_particles_for_partition.reset(new partsize_t[current_partition_size+1]);

        nb_processes_involved = nb_processes;
        while(nb_processes_involved != 0 && partition_interval_size_per_proc[nb_processes_involved-1] == 0){
            nb_processes_involved -= 1;
        }
        assert(nb_processes_involved != 0);
        for(int idx_proc_involved = 0 ; idx_proc_involved < nb_processes_involved ; ++idx_proc_involved){
            assert(partition_interval_size_per_proc[idx_proc_involved] != 0);
        }

        assert(int(field_grid_dim[IDX_Z]) == partition_interval_offset_per_proc[nb_processes_involved]);
    }

    virtual ~particles_distr_mpi(){}

    ////////////////////////////////////////////////////////////////////////////

    template <class computer_class, class field_class, int size_particle_positions, int size_particle_rhs>
    void compute_distr(computer_class& in_computer,
                       field_class& in_field,
                       const partsize_t current_my_nb_particles_per_partition[],
                       const real_number particles_positions[],
                       real_number particles_current_rhs[],
                       const int interpolation_size){
        TIMEZONE("compute_distr");

        // Some processes might not be involved
        if(nb_processes_involved <= my_rank){
            return;
        }

        current_offset_particles_for_partition[0] = 0;
        partsize_t myTotalNbParticles = 0;
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
            for(int idxLower = 1 ; idxLower <= interpolation_size+1 ; idxLower += partition_interval_size_per_proc[nextDestProc]){
                nextDestProc = (nextDestProc-1+nb_processes_involved)%nb_processes_involved;
                if(nextDestProc == my_rank){
                    // We are back on our process
                    break;
                }

                const int destProc = nextDestProc;
                const int lowerRankDiff = (nextDestProc < my_rank ? my_rank - nextDestProc : nb_processes_involved-nextDestProc+my_rank);

                const int nbPartitionsToSend = std::min(current_partition_size, interpolation_size-(idxLower-1));
                assert(nbPartitionsToSend >= 0);
                const partsize_t nbParticlesToSend = current_offset_particles_for_partition[nbPartitionsToSend] - current_offset_particles_for_partition[0];

                const int nbPartitionsToRecv = std::min(partition_interval_size_per_proc[destProc], (interpolation_size+1)-(idxLower-1));
                assert(nbPartitionsToRecv > 0);
                const partsize_t nbParticlesToRecv = -1;

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
            nbProcToRecvLower = int(neigDescriptors.size());

            nextDestProc = my_rank;
            for(int idxUpper = 1 ; idxUpper <= interpolation_size+1 ; idxUpper += partition_interval_size_per_proc[nextDestProc]){
                nextDestProc = (nextDestProc+1+nb_processes_involved)%nb_processes_involved;
                if(nextDestProc == my_rank){
                    // We are back on our process
                    break;
                }

                const int destProc = nextDestProc;
                const int upperRankDiff = (nextDestProc > my_rank ? nextDestProc - my_rank: nb_processes_involved-my_rank+nextDestProc);

                const int nbPartitionsToSend = std::min(current_partition_size, (interpolation_size+1)-(idxUpper-1));
                assert(nbPartitionsToSend > 0);
                const partsize_t nbParticlesToSend = current_offset_particles_for_partition[current_partition_size] - current_offset_particles_for_partition[current_partition_size-nbPartitionsToSend];

                const int nbPartitionsToRecv = std::min(partition_interval_size_per_proc[destProc], interpolation_size-(idxUpper-1));
                assert(nbPartitionsToSend >= 0);
                const partsize_t nbParticlesToRecv = -1;

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
        const int nbProcToRecvUpper = int(neigDescriptors.size())-nbProcToRecvLower;
        const int nbProcToRecv = nbProcToRecvUpper + nbProcToRecvLower;
        assert(int(neigDescriptors.size()) == nbProcToRecv);

        for(int idxDescr = 0 ; idxDescr < int(neigDescriptors.size()) ; ++idxDescr){
            NeighborDescriptor& descriptor = neigDescriptors[idxDescr];

            if(descriptor.isLower){
                if(descriptor.nbPartitionsToSend > 0){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Isend(const_cast<partsize_t*>(&descriptor.nbParticlesToSend), 1, particles_utils::GetMpiType(partsize_t()),
                                        descriptor.destProc, TAG_LOW_UP_NB_PARTICLES,
                                        current_com, &mpiRequests.back()));

                    if(descriptor.nbParticlesToSend){
                        whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                        mpiRequests.emplace_back();
                        assert(descriptor.nbParticlesToSend*size_particle_positions < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Isend(const_cast<real_number*>(&particles_positions[0]), int(descriptor.nbParticlesToSend*size_particle_positions), particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_LOW_UP_PARTICLES,
                                  current_com, &mpiRequests.back()));

                        assert(descriptor.toRecvAndMerge == nullptr);
                        descriptor.toRecvAndMerge.reset(new real_number[descriptor.nbParticlesToSend*size_particle_rhs]);
                        whatNext.emplace_back(std::pair<Action,int>{MERGE_PARTICLES, idxDescr});
                        mpiRequests.emplace_back();
                        assert(descriptor.nbParticlesToSend*size_particle_rhs < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Irecv(descriptor.toRecvAndMerge.get(), int(descriptor.nbParticlesToSend*size_particle_rhs), particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_UP_LOW_RESULTS,
                                  current_com, &mpiRequests.back()));
                    }
                }

                assert(descriptor.nbPartitionsToRecv);
                whatNext.emplace_back(std::pair<Action,int>{RECV_PARTICLES, idxDescr});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Irecv(&descriptor.nbParticlesToRecv,
                          1, particles_utils::GetMpiType(partsize_t()), descriptor.destProc, TAG_UP_LOW_NB_PARTICLES,
                          current_com, &mpiRequests.back()));
            }
            else{
                assert(descriptor.nbPartitionsToSend);
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                AssertMpi(MPI_Isend(const_cast<partsize_t*>(&descriptor.nbParticlesToSend), 1, particles_utils::GetMpiType(partsize_t()),
                                    descriptor.destProc, TAG_UP_LOW_NB_PARTICLES,
                                    current_com, &mpiRequests.back()));

                if(descriptor.nbParticlesToSend){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();                    
                    assert(descriptor.nbParticlesToSend*size_particle_positions < std::numeric_limits<int>::max());
                    AssertMpi(MPI_Isend(const_cast<real_number*>(&particles_positions[(current_offset_particles_for_partition[current_partition_size-descriptor.nbPartitionsToSend])*size_particle_positions]),
                                        int(descriptor.nbParticlesToSend*size_particle_positions), particles_utils::GetMpiType(real_number()),
                                        descriptor.destProc, TAG_UP_LOW_PARTICLES,
                                        current_com, &mpiRequests.back()));

                    assert(descriptor.toRecvAndMerge == nullptr);
                    descriptor.toRecvAndMerge.reset(new real_number[descriptor.nbParticlesToSend*size_particle_rhs]);
                    whatNext.emplace_back(std::pair<Action,int>{MERGE_PARTICLES, idxDescr});
                    mpiRequests.emplace_back();
                    assert(descriptor.nbParticlesToSend*size_particle_rhs < std::numeric_limits<int>::max());
                    AssertMpi(MPI_Irecv(descriptor.toRecvAndMerge.get(), int(descriptor.nbParticlesToSend*size_particle_rhs), particles_utils::GetMpiType(real_number()), descriptor.destProc, TAG_LOW_UP_RESULTS,
                              current_com, &mpiRequests.back()));
                }

                if(descriptor.nbPartitionsToRecv){
                    whatNext.emplace_back(std::pair<Action,int>{RECV_PARTICLES, idxDescr});
                    mpiRequests.emplace_back();
                    AssertMpi(MPI_Irecv(&descriptor.nbParticlesToRecv,
                          1, particles_utils::GetMpiType(partsize_t()), descriptor.destProc, TAG_LOW_UP_NB_PARTICLES,
                          current_com, &mpiRequests.back()));
                }
            }
        }

        const bool more_than_one_thread = (omp_get_max_threads() > 1);

        TIMEZONE_OMP_INIT_PREPARALLEL(omp_get_max_threads())
        #pragma omp parallel default(shared)
        {
            #pragma omp master
            {
                while(mpiRequests.size()){
                    assert(mpiRequests.size() == whatNext.size());

                    int idxDone = int(mpiRequests.size());
                    {
                        TIMEZONE("wait");
                        AssertMpi(MPI_Waitany(int(mpiRequests.size()), mpiRequests.data(), &idxDone, MPI_STATUSES_IGNORE));
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
                        NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];

                        if(descriptor.isLower){
                            //const int idxLower = descriptor.idxLowerUpper;
                            const int destProc = descriptor.destProc;
                            //const int nbPartitionsToRecv = descriptor.nbPartitionsToRecv;
                            const partsize_t NbParticlesToReceive = descriptor.nbParticlesToRecv;
                            assert(NbParticlesToReceive != -1);
                            assert(descriptor.toCompute == nullptr);
                            if(NbParticlesToReceive){
                                descriptor.toCompute.reset(new real_number[NbParticlesToReceive*size_particle_positions]);
                                whatNext.emplace_back(std::pair<Action,int>{COMPUTE_PARTICLES, releasedAction.second});
                                mpiRequests.emplace_back();
                                assert(NbParticlesToReceive*size_particle_positions < std::numeric_limits<int>::max());
                                AssertMpi(MPI_Irecv(descriptor.toCompute.get(), int(NbParticlesToReceive*size_particle_positions),
                                                    particles_utils::GetMpiType(real_number()), destProc, TAG_UP_LOW_PARTICLES,
                                                    current_com, &mpiRequests.back()));
                            }
                        }
                        else{
                            //const int idxUpper = descriptor.idxLowerUpper;
                            const int destProc = descriptor.destProc;
                            //const int nbPartitionsToRecv = descriptor.nbPartitionsToRecv;
                            const partsize_t NbParticlesToReceive = descriptor.nbParticlesToRecv;
                            assert(NbParticlesToReceive != -1);
                            assert(descriptor.toCompute == nullptr);
                            if(NbParticlesToReceive){
                                descriptor.toCompute.reset(new real_number[NbParticlesToReceive*size_particle_positions]);
                                whatNext.emplace_back(std::pair<Action,int>{COMPUTE_PARTICLES, releasedAction.second});
                                mpiRequests.emplace_back();
                                assert(NbParticlesToReceive*size_particle_positions < std::numeric_limits<int>::max());
                                AssertMpi(MPI_Irecv(descriptor.toCompute.get(), int(NbParticlesToReceive*size_particle_positions),
                                                    particles_utils::GetMpiType(real_number()), destProc, TAG_LOW_UP_PARTICLES,
                                                    current_com, &mpiRequests.back()));
                            }
                        }
                    }

                    //////////////////////////////////////////////////////////////////////
                    /// Computation
                    //////////////////////////////////////////////////////////////////////
                    if(releasedAction.first == COMPUTE_PARTICLES){
                        NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];
                        const partsize_t NbParticlesToReceive = descriptor.nbParticlesToRecv;

                        assert(descriptor.toCompute != nullptr);
                        descriptor.results.reset(new real_number[NbParticlesToReceive*size_particle_rhs]);
                        in_computer.template init_result_array<size_particle_rhs>(descriptor.results.get(), NbParticlesToReceive);

                        if(more_than_one_thread == false){
                            in_computer.template apply_computation<field_class, size_particle_rhs>(in_field, descriptor.toCompute.get(), descriptor.results.get(), NbParticlesToReceive);
                        }
                        else{
                            TIMEZONE_OMP_INIT_PRETASK(timeZoneTaskKey)
                            NeighborDescriptor* ptr_descriptor = &descriptor;
                            #pragma omp taskgroup
                            {
                                for(partsize_t idxPart = 0 ; idxPart < NbParticlesToReceive ; idxPart += 300){
                                    const partsize_t sizeToDo = std::min(partsize_t(300), NbParticlesToReceive-idxPart);
                                    #pragma omp task default(shared) firstprivate(ptr_descriptor, idxPart, sizeToDo) priority(10) \
                                             TIMEZONE_OMP_PRAGMA_TASK_KEY(timeZoneTaskKey)
                                    {
                                        TIMEZONE_OMP_TASK("in_computer.apply_computation", timeZoneTaskKey);
                                        in_computer.template apply_computation<field_class, size_particle_rhs>(in_field, &ptr_descriptor->toCompute[idxPart*size_particle_positions],
                                                &ptr_descriptor->results[idxPart*size_particle_rhs], sizeToDo);
                                    }
                                }
                            }
                        }

                        const int destProc = descriptor.destProc;
                        whatNext.emplace_back(std::pair<Action,int>{RELEASE_BUFFER_PARTICLES, releasedAction.second});
                        mpiRequests.emplace_back();
                        const int tag = descriptor.isLower? TAG_LOW_UP_RESULTS : TAG_UP_LOW_RESULTS;                        
                        assert(NbParticlesToReceive*size_particle_rhs < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Isend(descriptor.results.get(), int(NbParticlesToReceive*size_particle_rhs), particles_utils::GetMpiType(real_number()), destProc, tag,
                                  current_com, &mpiRequests.back()));
                    }
                    //////////////////////////////////////////////////////////////////////
                    /// Computation
                    //////////////////////////////////////////////////////////////////////
                    if(releasedAction.first == RELEASE_BUFFER_PARTICLES){
                        NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];
                        assert(descriptor.toCompute != nullptr);
                        descriptor.toCompute.release();
                    }
                    //////////////////////////////////////////////////////////////////////
                    /// Merge
                    //////////////////////////////////////////////////////////////////////
                    if(releasedAction.first == MERGE_PARTICLES && more_than_one_thread == false){
                        NeighborDescriptor& descriptor = neigDescriptors[releasedAction.second];

                        if(descriptor.isLower){
                            TIMEZONE("reduce");
                            assert(descriptor.toRecvAndMerge != nullptr);
                            in_computer.template reduce_particles_rhs<size_particle_rhs>(&particles_current_rhs[0], descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                            descriptor.toRecvAndMerge.release();
                        }
                        else {
                            TIMEZONE("reduce");
                            assert(descriptor.toRecvAndMerge != nullptr);
                            in_computer.template reduce_particles_rhs<size_particle_rhs>(&particles_current_rhs[(current_offset_particles_for_partition[current_partition_size]-descriptor.nbParticlesToSend)*size_particle_rhs],
                                             descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                            descriptor.toRecvAndMerge.release();
                        }
                    }
                }
            }
            if(more_than_one_thread && omp_get_thread_num() == 1){
                TIMEZONE_OMP_INIT_PRETASK(timeZoneTaskKey)
                #pragma omp taskgroup
                {
                    // Do for all partitions except the first and last one
                    for(int idxPartition = 0 ; idxPartition < current_partition_size ; ++idxPartition){
                        for(partsize_t idxPart = current_offset_particles_for_partition[idxPartition] ;
                            idxPart < current_offset_particles_for_partition[idxPartition+1] ; idxPart += 300){

                            const partsize_t sizeToDo = std::min(partsize_t(300), current_offset_particles_for_partition[idxPartition+1]-idxPart);

                            // Low priority to help master thread when possible
                            #pragma omp task default(shared) firstprivate(idxPart, sizeToDo) priority(0) TIMEZONE_OMP_PRAGMA_TASK_KEY(timeZoneTaskKey)
                            {
                                TIMEZONE_OMP_TASK("in_computer.apply_computation", timeZoneTaskKey);
                                in_computer.template apply_computation<field_class, size_particle_rhs>(in_field, &particles_positions[idxPart*size_particle_positions],
                                                  &particles_current_rhs[idxPart*size_particle_rhs],
                                                  sizeToDo);
                            }
                        }
                    }
                }
            }
        }

        if(more_than_one_thread == true){
            for(int idxDescr = 0 ; idxDescr < int(neigDescriptors.size()) ; ++idxDescr){
                NeighborDescriptor& descriptor = neigDescriptors[idxDescr];
                if(descriptor.nbParticlesToSend){
                    if(descriptor.isLower){
                        TIMEZONE("reduce_later");
                        assert(descriptor.toRecvAndMerge != nullptr);
                        in_computer.template reduce_particles_rhs<size_particle_rhs>(&particles_current_rhs[0], descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                        descriptor.toRecvAndMerge.release();
                    }
                    else {
                        TIMEZONE("reduce_later");
                        assert(descriptor.toRecvAndMerge != nullptr);
                        in_computer.template reduce_particles_rhs<size_particle_rhs>(&particles_current_rhs[(current_offset_particles_for_partition[current_partition_size]-descriptor.nbParticlesToSend)*size_particle_rhs],
                                         descriptor.toRecvAndMerge.get(), descriptor.nbParticlesToSend);
                        descriptor.toRecvAndMerge.release();
                    }
                }
            }
        }

        // Do my own computation if not threaded
        if(more_than_one_thread == false){
            TIMEZONE("compute-my_compute");
            // Compute my particles
            if(myTotalNbParticles){
                in_computer.template apply_computation<field_class, size_particle_rhs>(in_field, particles_positions, particles_current_rhs, myTotalNbParticles);
            }
        }

        assert(whatNext.size() == 0);
        assert(mpiRequests.size() == 0);
    }


    ////////////////////////////////////////////////////////////////////////////

    template <class computer_class, int size_particle_positions, int size_particle_rhs, int size_particle_index>
    void redistribute(computer_class& in_computer,
                      partsize_t current_my_nb_particles_per_partition[],
                      partsize_t* nb_particles,
                      std::unique_ptr<real_number[]>* inout_positions_particles,
                      std::unique_ptr<real_number[]> inout_rhs_particles[], const int in_nb_rhs,
                      std::unique_ptr<partsize_t[]>* inout_index_particles){
        TIMEZONE("redistribute");

        // Some latest processes might not be involved
        if(nb_processes_involved <= my_rank){
            return;
        }

        current_offset_particles_for_partition[0] = 0;
        partsize_t myTotalNbParticles = 0;
        for(int idxPartition = 0 ; idxPartition < current_partition_size ; ++idxPartition){
            myTotalNbParticles += current_my_nb_particles_per_partition[idxPartition];
            current_offset_particles_for_partition[idxPartition+1] = current_offset_particles_for_partition[idxPartition] + current_my_nb_particles_per_partition[idxPartition];
        }
        assert((*nb_particles) == myTotalNbParticles);

        // Find particles outside my interval
        const partsize_t nbOutLower = particles_utils::partition_extra<partsize_t, size_particle_positions>(&(*inout_positions_particles)[0], current_my_nb_particles_per_partition[0],
                    [&](const real_number val[]){
            const int partition_level = in_computer.pbc_field_layer(val[IDX_Z], IDX_Z);
            assert(partition_level == current_partition_interval.first
                   || partition_level == (current_partition_interval.first-1+int(field_grid_dim[IDX_Z]))%int(field_grid_dim[IDX_Z])
                   || partition_level == (current_partition_interval.first+1)%int(field_grid_dim[IDX_Z]));
            const bool isLower = partition_level == (current_partition_interval.first-1+int(field_grid_dim[IDX_Z]))%int(field_grid_dim[IDX_Z]);
            return isLower;
        },
                    [&](const partsize_t idx1, const partsize_t idx2){
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
        const partsize_t offesetOutLow = (current_partition_size==1? nbOutLower : 0);

        const partsize_t nbOutUpper = current_my_nb_particles_per_partition[current_partition_size-1] - offesetOutLow - particles_utils::partition_extra<partsize_t, size_particle_positions>(
                    &(*inout_positions_particles)[(current_offset_particles_for_partition[current_partition_size-1]+offesetOutLow)*size_particle_positions],
                    myTotalNbParticles - (current_offset_particles_for_partition[current_partition_size-1]+offesetOutLow),
                    [&](const real_number val[]){
            const int partition_level = in_computer.pbc_field_layer(val[IDX_Z], IDX_Z);
            assert(partition_level == (current_partition_interval.second-1)
                   || partition_level == ((current_partition_interval.second-1)-1+int(field_grid_dim[IDX_Z]))%int(field_grid_dim[IDX_Z])
                   || partition_level == ((current_partition_interval.second-1)+1)%int(field_grid_dim[IDX_Z]));
            const bool isUpper = (partition_level == ((current_partition_interval.second-1)+1)%int(field_grid_dim[IDX_Z]));
            return !isUpper;
        },
                    [&](const partsize_t idx1, const partsize_t idx2){
            for(int idx_val = 0 ; idx_val < size_particle_index ; ++idx_val){
                std::swap((*inout_index_particles)[idx1], (*inout_index_particles)[idx2]);
            }

            for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                for(int idx_val = 0 ; idx_val < size_particle_rhs ; ++idx_val){
                    std::swap(inout_rhs_particles[idx_rhs][idx1*size_particle_rhs + idx_val],
                              inout_rhs_particles[idx_rhs][idx2*size_particle_rhs + idx_val]);
                }
            }
        }, (current_offset_particles_for_partition[current_partition_size-1]+offesetOutLow));

        // Exchange number
        int eventsBeforeWaitall = 0;
        partsize_t nbNewFromLow = 0;
        partsize_t nbNewFromUp = 0;
        std::unique_ptr<real_number[]> newParticlesLow;
        std::unique_ptr<real_number[]> newParticlesUp;
        std::unique_ptr<partsize_t[]> newParticlesLowIndexes;
        std::unique_ptr<partsize_t[]> newParticlesUpIndexes;
        std::vector<std::unique_ptr<real_number[]>> newParticlesLowRhs(in_nb_rhs);
        std::vector<std::unique_ptr<real_number[]>> newParticlesUpRhs(in_nb_rhs);

        {
            assert(whatNext.size() == 0);
            assert(mpiRequests.size() == 0);

            whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_NB_LOW, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Irecv(&nbNewFromLow, 1, particles_utils::GetMpiType(partsize_t()),
                                (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_UP_LOW_MOVED_NB_PARTICLES,
                                MPI_COMM_WORLD, &mpiRequests.back()));
            eventsBeforeWaitall += 1;

            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Isend(const_cast<partsize_t*>(&nbOutLower), 1, particles_utils::GetMpiType(partsize_t()),
                                (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_LOW_UP_MOVED_NB_PARTICLES,
                                MPI_COMM_WORLD, &mpiRequests.back()));

            if(nbOutLower){
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();                
                assert(nbOutLower*size_particle_positions < std::numeric_limits<int>::max());
                AssertMpi(MPI_Isend(&(*inout_positions_particles)[0], int(nbOutLower*size_particle_positions), particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES,
                          MPI_COMM_WORLD, &mpiRequests.back()));
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                assert(nbOutLower < std::numeric_limits<int>::max());
                AssertMpi(MPI_Isend(&(*inout_index_particles)[0], int(nbOutLower), particles_utils::GetMpiType(partsize_t()),
                          (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES_INDEXES,
                          MPI_COMM_WORLD, &mpiRequests.back()));

                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    assert(nbOutLower*size_particle_rhs < std::numeric_limits<int>::max());
                    AssertMpi(MPI_Isend(&inout_rhs_particles[idx_rhs][0], int(nbOutLower*size_particle_rhs), particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES_RHS+idx_rhs,
                              MPI_COMM_WORLD, &mpiRequests.back()));
                }
            }

            whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_NB_UP, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Irecv(&nbNewFromUp, 1, particles_utils::GetMpiType(partsize_t()), (my_rank+1)%nb_processes_involved,
                                TAG_LOW_UP_MOVED_NB_PARTICLES,
                                MPI_COMM_WORLD, &mpiRequests.back()));
            eventsBeforeWaitall += 1;

            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
            mpiRequests.emplace_back();
            AssertMpi(MPI_Isend(const_cast<partsize_t*>(&nbOutUpper), 1, particles_utils::GetMpiType(partsize_t()),
                                (my_rank+1)%nb_processes_involved, TAG_UP_LOW_MOVED_NB_PARTICLES,
                                MPI_COMM_WORLD, &mpiRequests.back()));

            if(nbOutUpper){
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                assert(nbOutUpper*size_particle_positions < std::numeric_limits<int>::max());
                AssertMpi(MPI_Isend(&(*inout_positions_particles)[(myTotalNbParticles-nbOutUpper)*size_particle_positions],
                          int(nbOutUpper*size_particle_positions), particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES,
                          MPI_COMM_WORLD, &mpiRequests.back()));
                whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                mpiRequests.emplace_back();
                assert(nbOutUpper < std::numeric_limits<int>::max());
                AssertMpi(MPI_Isend(&(*inout_index_particles)[(myTotalNbParticles-nbOutUpper)], int(nbOutUpper),
                          particles_utils::GetMpiType(partsize_t()), (my_rank+1)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES_INDEXES,
                          MPI_COMM_WORLD, &mpiRequests.back()));


                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                    mpiRequests.emplace_back();
                    assert(nbOutUpper*size_particle_rhs < std::numeric_limits<int>::max());
                    AssertMpi(MPI_Isend(&inout_rhs_particles[idx_rhs][(myTotalNbParticles-nbOutUpper)*size_particle_rhs],
                              int(nbOutUpper*size_particle_rhs), particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES_RHS+idx_rhs,
                              MPI_COMM_WORLD, &mpiRequests.back()));
                }
            }

            while(mpiRequests.size() && eventsBeforeWaitall){
                int idxDone = int(mpiRequests.size());
                {
                    TIMEZONE("waitany_move");
                    AssertMpi(MPI_Waitany(int(mpiRequests.size()), mpiRequests.data(), &idxDone, MPI_STATUSES_IGNORE));
                }
                const std::pair<Action, int> releasedAction = whatNext[idxDone];
                std::swap(mpiRequests[idxDone], mpiRequests[mpiRequests.size()-1]);
                std::swap(whatNext[idxDone], whatNext[mpiRequests.size()-1]);
                mpiRequests.pop_back();
                whatNext.pop_back();

                if(releasedAction.first == RECV_MOVE_NB_LOW){
                    if(nbNewFromLow){
                        assert(newParticlesLow == nullptr);
                        newParticlesLow.reset(new real_number[nbNewFromLow*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_LOW, -1});
                        mpiRequests.emplace_back();
                        assert(nbNewFromLow*size_particle_positions < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Irecv(&newParticlesLow[0], int(nbNewFromLow*size_particle_positions), particles_utils::GetMpiType(real_number()),
                                  (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        newParticlesLowIndexes.reset(new partsize_t[nbNewFromLow]);
                        whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                        mpiRequests.emplace_back();
                        assert(nbNewFromLow < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Irecv(&newParticlesLowIndexes[0], int(nbNewFromLow), particles_utils::GetMpiType(partsize_t()),
                                  (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES_INDEXES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            newParticlesLowRhs[idx_rhs].reset(new real_number[nbNewFromLow*size_particle_rhs]);
                            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                            mpiRequests.emplace_back();
                            assert(nbNewFromLow*size_particle_rhs < std::numeric_limits<int>::max());
                            AssertMpi(MPI_Irecv(&newParticlesLowRhs[idx_rhs][0], int(nbNewFromLow*size_particle_rhs), particles_utils::GetMpiType(real_number()), (my_rank-1+nb_processes_involved)%nb_processes_involved, TAG_UP_LOW_MOVED_PARTICLES_RHS+idx_rhs,
                                      MPI_COMM_WORLD, &mpiRequests.back()));
                        }
                    }
                    eventsBeforeWaitall -= 1;
                }
                else if(releasedAction.first == RECV_MOVE_NB_UP){
                    if(nbNewFromUp){
                        assert(newParticlesUp == nullptr);
                        newParticlesUp.reset(new real_number[nbNewFromUp*size_particle_positions]);
                        whatNext.emplace_back(std::pair<Action,int>{RECV_MOVE_UP, -1});
                        mpiRequests.emplace_back();
                        assert(nbNewFromUp*size_particle_positions < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Irecv(&newParticlesUp[0], int(nbNewFromUp*size_particle_positions), particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        newParticlesUpIndexes.reset(new partsize_t[nbNewFromUp]);
                        whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                        mpiRequests.emplace_back();
                        assert(nbNewFromUp < std::numeric_limits<int>::max());
                        AssertMpi(MPI_Irecv(&newParticlesUpIndexes[0], int(nbNewFromUp), particles_utils::GetMpiType(partsize_t()),
                                  (my_rank+1)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES_INDEXES,
                                  MPI_COMM_WORLD, &mpiRequests.back()));

                        for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                            newParticlesUpRhs[idx_rhs].reset(new real_number[nbNewFromUp*size_particle_rhs]);
                            whatNext.emplace_back(std::pair<Action,int>{NOTHING_TODO, -1});
                            mpiRequests.emplace_back();
                            assert(nbNewFromUp*size_particle_rhs < std::numeric_limits<int>::max());
                            AssertMpi(MPI_Irecv(&newParticlesUpRhs[idx_rhs][0], int(nbNewFromUp*size_particle_rhs), particles_utils::GetMpiType(real_number()), (my_rank+1)%nb_processes_involved, TAG_LOW_UP_MOVED_PARTICLES_RHS+idx_rhs,
                                      MPI_COMM_WORLD, &mpiRequests.back()));
                        }
                    }
                    eventsBeforeWaitall -= 1;
                }
            }

            if(mpiRequests.size()){
                // TODO Proceed when received
                TIMEZONE("waitall-move");
                AssertMpi(MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE));
                mpiRequests.clear();
                whatNext.clear();
            }
        }

        // Realloc an merge
        {
            TIMEZONE("realloc_copy");
            const partsize_t nbOldParticlesInside = myTotalNbParticles - nbOutLower - nbOutUpper;
            const partsize_t myTotalNewNbParticles = nbOldParticlesInside + nbNewFromLow + nbNewFromUp;

            std::unique_ptr<real_number[]> newArray(new real_number[myTotalNewNbParticles*size_particle_positions]);
            std::unique_ptr<partsize_t[]> newArrayIndexes(new partsize_t[myTotalNewNbParticles]);
            std::vector<std::unique_ptr<real_number[]>> newArrayRhs(in_nb_rhs);
            for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                newArrayRhs[idx_rhs].reset(new real_number[myTotalNewNbParticles*size_particle_rhs]);
            }

            // Copy new particles recv form lower first
            if(nbNewFromLow){
                const particles_utils::fixed_copy fcp(0, 0, nbNewFromLow);
                fcp.copy(newArray, newParticlesLow, size_particle_positions);
                fcp.copy(newArrayIndexes, newParticlesLowIndexes);
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    fcp.copy(newArrayRhs[idx_rhs], newParticlesLowRhs[idx_rhs], size_particle_rhs);
                }
            }

            // Copy my own particles
            {
                const particles_utils::fixed_copy fcp(nbNewFromLow, nbOutLower, nbOldParticlesInside);
                fcp.copy(newArray, (*inout_positions_particles), size_particle_positions);
                fcp.copy(newArrayIndexes, (*inout_index_particles));
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    fcp.copy(newArrayRhs[idx_rhs], inout_rhs_particles[idx_rhs], size_particle_rhs);
                }
            }

            // Copy new particles from upper at the back
            if(nbNewFromUp){
                const particles_utils::fixed_copy fcp(nbNewFromLow+nbOldParticlesInside, 0, nbNewFromUp);
                fcp.copy(newArray, newParticlesUp, size_particle_positions);
                fcp.copy(newArrayIndexes, newParticlesUpIndexes);
                for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                    fcp.copy(newArrayRhs[idx_rhs], newParticlesUpRhs[idx_rhs], size_particle_rhs);
                }
            }

            (*inout_positions_particles) = std::move(newArray);
            (*inout_index_particles) = std::move(newArrayIndexes);
            for(int idx_rhs = 0 ; idx_rhs < in_nb_rhs ; ++idx_rhs){
                inout_rhs_particles[idx_rhs] = std::move(newArrayRhs[idx_rhs]);
            }

            myTotalNbParticles = myTotalNewNbParticles;
        }

        // Partitions all particles
        {
            TIMEZONE("repartition");
            particles_utils::partition_extra_z<partsize_t, size_particle_positions>(&(*inout_positions_particles)[0],
                                             myTotalNbParticles,current_partition_size,
                                             current_my_nb_particles_per_partition, current_offset_particles_for_partition.get(),
                                             [&](const real_number& z_pos){
                const int partition_level = in_computer.pbc_field_layer(z_pos, IDX_Z);
                assert(current_partition_interval.first <= partition_level && partition_level < current_partition_interval.second);
                return partition_level - current_partition_interval.first;
            },
            [&](const partsize_t idx1, const partsize_t idx2){
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
                    for(partsize_t idx = current_offset_particles_for_partition[idxPartition] ; idx < current_offset_particles_for_partition[idxPartition+1] ; ++idx){
                        assert(in_computer.pbc_field_layer((*inout_positions_particles)[idx*3+IDX_Z], IDX_Z)-current_partition_interval.first == idxPartition);
                    }
                }
            }
        }
        (*nb_particles) = myTotalNbParticles;

        assert(mpiRequests.size() == 0);
    }
};

#endif
