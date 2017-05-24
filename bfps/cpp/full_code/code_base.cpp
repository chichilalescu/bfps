#include "code_base.hpp"
#include "scope_timer.hpp"

code_base::code_base(
        const MPI_Comm COMMUNICATOR,
        const std::string &simulation_name):
    comm(COMMUNICATOR),
    simname(simulation_name)
{
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
    this->stop_code_now = false;
}

int code_base::check_stopping_condition(void)
{
    if (myrank == 0)
    {
        std::string fname = (
                std::string("stop_") +
                std::string(this->simname));
        {
            struct stat file_buffer;
            this->stop_code_now = (
                    stat(fname.c_str(), &file_buffer) == 0);
        }
    }
    MPI_Bcast(
            &this->stop_code_now,
            1,
            MPI_C_BOOL,
            0,
            MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

