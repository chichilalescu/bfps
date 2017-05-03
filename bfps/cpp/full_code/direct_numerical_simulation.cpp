#include "direct_numerical_simulation.hpp"

direct_numerical_simulation::direct_numerical_simulation(
        const MPI_Comm COMMUNICATOR,
        const std::string &simulation_name):
    comm(COMMUNICATOR),
    simname(simulation_name)
{
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
}

