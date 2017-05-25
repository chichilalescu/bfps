#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "direct_numerical_simulation.hpp"
#include "scope_timer.hpp"
#include "hdf5_tools.hpp"


int direct_numerical_simulation::grow_file_datasets()
{
    return hdf5_tools::grow_file_datasets(
            this->stat_file,
            "statistics",
            this->niter_todo / this->niter_stat);
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
    this->start_simple_timer();
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
        this->print_simple_timer(
                "iteration " + std::to_string(this->iteration));
    }
    this->do_stats();
    this->print_simple_timer(
            "final call to do_stats ");
    if (this->iteration % this->niter_out != 0)
        this->write_checkpoint();
    return EXIT_SUCCESS;
}

