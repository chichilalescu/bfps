#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "direct_numerical_simulation.hpp"
#include "scope_timer.hpp"

int grow_single_dataset(hid_t dset, int tincrement)
{
    int ndims;
    hsize_t space;
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_ndims(space);
    hsize_t *dims = new hsize_t[ndims];
    H5Sget_simple_extent_dims(space, dims, NULL);
    dims[0] += tincrement;
    H5Dset_extent(dset, dims);
    H5Sclose(space);
    delete[] dims;
    return EXIT_SUCCESS;
}

herr_t grow_dataset_visitor(
    hid_t o_id,
    const char *name,
    const H5O_info_t *info,
    void *op_data)
{
    if (info->type == H5O_TYPE_DATASET)
    {
        hsize_t dset = H5Dopen(o_id, name, H5P_DEFAULT);
        grow_single_dataset(dset, *((int*)(op_data)));
        H5Dclose(dset);
    }
    return EXIT_SUCCESS;
}

direct_numerical_simulation::direct_numerical_simulation(
        const MPI_Comm COMMUNICATOR,
        const std::string &simulation_name):
    comm(COMMUNICATOR),
    simname(simulation_name)
{
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
    this->stop_code_now = false;
}


int direct_numerical_simulation::grow_file_datasets()
{
    int file_problems = 0;

    hid_t group;
    group = H5Gopen(this->stat_file, "/statistics", H5P_DEFAULT);
    int tincrement = this->niter_todo / this->niter_stat;
    H5Ovisit(
            group,
            H5_INDEX_NAME,
            H5_ITER_NATIVE,
            grow_dataset_visitor,
            &tincrement);
    H5Gclose(group);
    return file_problems;
}

int direct_numerical_simulation::read_iteration(void)
{
    /* read iteration */
    hid_t dset;
    hid_t iteration_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    dset = H5Dopen(
            iteration_file,
            "iteration",
            H5P_DEFAULT);
    H5Dread(
            dset,
            H5T_NATIVE_INT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->iteration);
    H5Dclose(dset);
    dset = H5Dopen(
            iteration_file,
            "checkpoint",
            H5P_DEFAULT);
    H5Dread(
            dset,
            H5T_NATIVE_INT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &this->checkpoint);
    H5Dclose(dset);
    H5Fclose(iteration_file);
    DEBUG_MSG("simname is %s, iteration is %d and checkpoint is %d\n",
            this->simname.c_str(),
            this->iteration,
            this->checkpoint);
    return EXIT_SUCCESS;
}

int direct_numerical_simulation::write_iteration(void)
{
    if (this->myrank == 0)
    {
        hid_t dset = H5Dopen(
                this->stat_file,
                "iteration",
                H5P_DEFAULT);
        H5Dwrite(
                dset,
                H5T_NATIVE_INT,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                &this->iteration);
        H5Dclose(dset);
        dset = H5Dopen(
                this->stat_file,
                "checkpoint",
                H5P_DEFAULT);
        H5Dwrite(
                dset,
                H5T_NATIVE_INT,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                &this->checkpoint);
        H5Dclose(dset);
    }
    return EXIT_SUCCESS;
}

int direct_numerical_simulation::main_loop(void)
{
    clock_t time0, time1;
    double time_difference, local_time_difference;
    time0 = clock();
    int max_iter = (this->iteration + this->niter_todo -
                    (this->iteration % this->niter_todo));
    for (; this->iteration < max_iter;)
    {
    #ifdef USE_TIMINGOUTPUT
        const std::string loopLabel = ("code::main_start::loop-" +
                                       std::to_string(this->iteration));
        TIMEZONE(loopLabel.c_str());
    #endif
        this->do_stats();

        this->step();
        if (this->iteration % this->niter_out == 0)
            this->write_checkpoint();
        this->check_stopping_condition();
        if (this->stop_code_now)
            break;
        time1 = clock();
        local_time_difference = ((
                (unsigned int)(time1 - time0)) /
                ((double)CLOCKS_PER_SEC));
        time_difference = 0.0;
        MPI_Allreduce(
                &local_time_difference,
                &time_difference,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
        if (this->myrank == 0)
            std::cout << "iteration " << iteration <<
                         " took " << time_difference/nprocs <<
                         " seconds" << std::endl;
        if (this->myrank == 0)
            std::cerr << "iteration " << iteration <<
                         " took " << time_difference/nprocs <<
                         " seconds" << std::endl;
        time0 = time1;
    }
    this->do_stats();
    time1 = clock();
    local_time_difference = ((
            (unsigned int)(time1 - time0)) /
            ((double)CLOCKS_PER_SEC));
    time_difference = 0.0;
    MPI_Allreduce(
            &local_time_difference,
            &time_difference,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);
    if (this->myrank == 0)
        std::cout << "iteration " << iteration <<
                     " took " << time_difference/nprocs <<
                     " seconds" << std::endl;
    if (this->myrank == 0)
        std::cerr << "iteration " << iteration <<
                     " took " << time_difference/nprocs <<
                     " seconds" << std::endl;
    if (this->iteration % this->niter_out != 0)
        this->write_checkpoint();
    return EXIT_SUCCESS;
}

int direct_numerical_simulation::check_stopping_condition(void)
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

