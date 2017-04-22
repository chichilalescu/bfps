#ifndef ALLTOALL_EXCHANGER_HPP
#define ALLTOALL_EXCHANGER_HPP

#include <mpi.h>
#include <cassert>

#include "base.hpp"
#include "particles_utils.hpp"
#include "scope_timer.hpp"

class alltoall_exchanger {
    const MPI_Comm mpi_com;

    int my_rank;
    int nb_processes;

    const std::vector<int> nb_items_to_send;

    std::vector<int> offset_items_to_send;

    std::vector<int> nb_items_to_sendrecv_all;
    std::vector<int> nb_items_to_recv;
    std::vector<int> offset_items_to_recv;

    int total_to_recv;

public:
    alltoall_exchanger(const MPI_Comm& in_mpi_com, std::vector<int>/*no ref to move here*/ in_nb_items_to_send)
        :mpi_com(in_mpi_com), nb_items_to_send(std::move(in_nb_items_to_send)), total_to_recv(0){
        TIMEZONE("alltoall_exchanger::constructor");

        AssertMpi(MPI_Comm_rank(mpi_com, &my_rank));
        AssertMpi(MPI_Comm_size(mpi_com, &nb_processes));

        assert(int(nb_items_to_send.size()) == nb_processes);

        offset_items_to_send.resize(nb_processes+1, 0);
        for(int idx_proc = 0 ; idx_proc < nb_processes ; ++idx_proc){
            offset_items_to_send[idx_proc+1] = offset_items_to_send[idx_proc]
                                             + nb_items_to_send[idx_proc];
        }

        nb_items_to_sendrecv_all.resize(nb_processes*nb_processes);
        AssertMpi(MPI_Allgather(const_cast<int*>(nb_items_to_send.data()), nb_processes, MPI_INT,
                          nb_items_to_sendrecv_all.data(), nb_processes, MPI_INT,
                          mpi_com));

        nb_items_to_recv.resize(nb_processes, 0);
        offset_items_to_recv.resize(nb_processes+1, 0);
        for(int idx_proc = 0 ; idx_proc < nb_processes ; ++idx_proc){
            const int nbrecv = nb_items_to_sendrecv_all[idx_proc*nb_processes + my_rank];
            total_to_recv += nbrecv;
            nb_items_to_recv[idx_proc] = nbrecv;
            offset_items_to_recv[idx_proc+1] = nb_items_to_recv[idx_proc]
                                                    + offset_items_to_recv[idx_proc];
        }
    }

    int getTotalToRecv() const{
        return total_to_recv;
    }

    template <class ItemType>
    void alltoallv_dt(const ItemType in_to_send[],
                   ItemType out_to_recv[], const MPI_Datatype& in_type) const {
        TIMEZONE("alltoallv");
        AssertMpi(MPI_Alltoallv(const_cast<ItemType*>(in_to_send), const_cast<int*>(nb_items_to_send.data()),
                          const_cast<int*>(offset_items_to_send.data()), in_type, out_to_recv,
                          const_cast<int*>(nb_items_to_recv.data()), const_cast<int*>(offset_items_to_recv.data()), in_type,
                          mpi_com));
    }

    template <class ItemType>
    void alltoallv(const ItemType in_to_send[],
                   ItemType out_to_recv[]) const {
        alltoallv_dt<ItemType>(in_to_send, out_to_recv, particles_utils::GetMpiType(ItemType()));
    }

    template <class ItemType>
    void alltoallv_dt(const ItemType in_to_send[],
                   ItemType out_to_recv[], const MPI_Datatype& in_type, const int in_nb_values_per_item) const {
        TIMEZONE("alltoallv");
        std::vector<int> nb_items_to_send_tmp = nb_items_to_send;
        particles_utils::transform(nb_items_to_send_tmp.begin(), nb_items_to_send_tmp.end(), nb_items_to_send_tmp.begin(),
                                   [&](const int val) -> int { return val * in_nb_values_per_item ;});
        std::vector<int> offset_items_to_send_tmp = offset_items_to_send;
        particles_utils::transform(offset_items_to_send_tmp.begin(), offset_items_to_send_tmp.end(), offset_items_to_send_tmp.begin(),
                                   [&](const int val) -> int { return val * in_nb_values_per_item ;});
        std::vector<int> nb_items_to_recv_tmp = nb_items_to_recv;
        particles_utils::transform(nb_items_to_recv_tmp.begin(), nb_items_to_recv_tmp.end(), nb_items_to_recv_tmp.begin(),
                                   [&](const int val) -> int { return val * in_nb_values_per_item ;});
        std::vector<int> offset_items_to_recv_tmp = offset_items_to_recv;
        particles_utils::transform(offset_items_to_recv_tmp.begin(), offset_items_to_recv_tmp.end(), offset_items_to_recv_tmp.begin(),
                                   [&](const int val) -> int { return val * in_nb_values_per_item ;});

        AssertMpi(MPI_Alltoallv(const_cast<ItemType*>(in_to_send), const_cast<int*>(nb_items_to_send_tmp.data()),
                          const_cast<int*>(offset_items_to_send_tmp.data()), in_type, out_to_recv,
                          const_cast<int*>(nb_items_to_recv_tmp.data()), const_cast<int*>(offset_items_to_recv_tmp.data()), in_type,
                          mpi_com));
    }

    template <class ItemType>
    void alltoallv(const ItemType in_to_send[],
                   ItemType out_to_recv[], const int in_nb_values_per_item) const {
        alltoallv_dt<ItemType>(in_to_send, out_to_recv,particles_utils::GetMpiType(ItemType()), in_nb_values_per_item);
    }
};

#endif
