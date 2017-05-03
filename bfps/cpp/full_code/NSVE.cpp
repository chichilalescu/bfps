#include <string>
#include <cmath>
#include "NSVE.hpp"

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

template <typename rnumber>
int NSVE<rnumber>::read_iteration(void)
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

template <typename rnumber>
int NSVE<rnumber>::write_iteration(void)
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

template <typename rnumber>
int NSVE<rnumber>::initialize(void)
{
    this->read_iteration();
    this->read_parameters();
    if (this->myrank == 0)
    {
        // set caching parameters
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        herr_t cache_err = H5Pset_cache(fapl, 0, 521, 134217728, 1.0);
        DEBUG_MSG("when setting stat_file cache I got %d\n", cache_err);
        this->stat_file = H5Fopen(
                (this->simname + ".h5").c_str(),
                H5F_ACC_RDWR,
                fapl);
    }
    int data_file_problem;
    if (this->myrank == 0)
        data_file_problem = this->grow_file_datasets();
    MPI_Bcast(&data_file_problem, 1, MPI_INT, 0, this->comm);
    if (data_file_problem > 0)
    {
        std::cerr <<
            data_file_problem <<
            " problems growing file datasets.\ntrying to exit now." <<
            std::endl;
        return EXIT_FAILURE;
    }
    this->fs = new vorticity_equation<rnumber, FFTW>(
            simname.c_str(),
            nx, ny, nz,
            dkx, dky, dkz,
            FFTW_MEASURE);
    this->tmp_vec_field = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            FFTW_MEASURE);


    this->fs->checkpoints_per_file = checkpoints_per_file;
    this->fs->nu = nu;
    this->fs->fmode = fmode;
    this->fs->famplitude = famplitude;
    this->fs->fk0 = fk0;
    this->fs->fk1 = fk1;
    strncpy(this->fs->forcing_type, forcing_type, 128);
    this->fs->iteration = this->iteration;
    this->fs->checkpoint = this->checkpoint;

    this->fs->cvorticity->real_space_representation = false;
    this->fs->io_checkpoint();

    if (this->myrank == 0 && this->iteration == 0)
        this->fs->kk->store(stat_file);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::main_loop(void)
{
    clock_t time0, time1;
    double time_difference, local_time_difference;
    time0 = clock();
    bool stop_code_now = false;
    int max_iter = (this->iteration + this->niter_todo -
                    (this->iteration % this->niter_todo));
    for (; this->iteration < max_iter;)
    {
    #ifdef USE_TIMINGOUTPUT
        const std::string loopLabel = ("code::main_start::loop-" +
                                       std::to_string(this->iteration));
        TIMEZONE(loopLabel.c_str());
    #endif
        if (this->iteration % this->niter_stat == 0)
            this->do_stats();
        this->fs->step(dt);
        this->iteration = this->fs->iteration;
        if (this->fs->iteration % this->niter_out == 0)
        {
            this->fs->io_checkpoint(false);
            this->checkpoint = this->fs->checkpoint;
            this->write_iteration();
        }
        if (stop_code_now)
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
            std::cout << "iteration " << iteration << " took " << time_difference/nprocs << " seconds" << std::endl;
        if (this->myrank == 0)
            std::cerr << "iteration " << iteration << " took " << time_difference/nprocs << " seconds" << std::endl;
        time0 = time1;
    }
    if (this->iteration % this->niter_stat == 0)
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
        std::cout << "iteration " << iteration << " took " << time_difference/nprocs << " seconds" << std::endl;
    if (this->myrank == 0)
        std::cerr << "iteration " << iteration << " took " << time_difference/nprocs << " seconds" << std::endl;
    if (this->iteration % this->niter_out != 0)
    {
        this->fs->io_checkpoint(false);
        this->checkpoint = this->fs->checkpoint;
        this->write_iteration();
    }
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVE<rnumber>::finalize(void)
{
    if (this->myrank == 0)
        H5Fclose(this->stat_file);
    delete this->fs;
    delete this->tmp_vec_field;
    return EXIT_SUCCESS;
}

//template <typename rnumber>
//int NSVE<rnumber>::read_parameters(void)
//{
//    hid_t parameter_file;
//    hid_t dset, memtype, space;
//    char *string_data;
//    parameter_file = H5Fopen(
//            (this->simname + std::string(".h5")).c_str(),
//            H5F_ACC_RDONLY,
//            H5P_DEFAULT);
//    dset = H5Dopen(parameter_file, "/parameters/checkpoints_per_file", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &checkpoints_per_file);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/dealias_type", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dealias_type);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/dkx", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dkx);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/dky", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dky);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/dkz", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dkz);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/dt", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dt);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/famplitude", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &famplitude);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/fk0", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fk0);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/fk1", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fk1);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/fmode", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fmode);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/forcing_type", H5P_DEFAULT);
//    space = H5Dget_space(dset);
//    memtype = H5Dget_type(dset);
//    string_data = (char*)malloc(256);
//    H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &string_data);
//    sprintf(forcing_type, "%s", string_data);
//    free(string_data);
//    H5Sclose(space);
//    H5Tclose(memtype);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/histogram_bins", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &histogram_bins);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/max_velocity_estimate", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &max_velocity_estimate);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/max_vorticity_estimate", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &max_vorticity_estimate);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/niter_out", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &niter_out);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/niter_part", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &niter_part);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/niter_stat", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &niter_stat);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/niter_todo", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &niter_todo);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/nparticles", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nparticles);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/nu", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nu);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/nx", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nx);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/ny", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ny);
//    H5Dclose(dset);
//    dset = H5Dopen(parameter_file, "/parameters/nz", H5P_DEFAULT);
//    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nz);
//    H5Dclose(dset);
//    H5Fclose(parameter_file);
//    return EXIT_SUCCESS;
//}


template <typename rnumber>
int NSVE<rnumber>::grow_file_datasets()
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

template <typename rnumber>
int NSVE<rnumber>::do_stats()
{
    hid_t stat_group;
    if (this->myrank == 0)
        stat_group = H5Gopen(
                this->stat_file,
                "statistics",
                H5P_DEFAULT);
    else
        stat_group = 0;
    fs->compute_velocity(fs->cvorticity);
    *tmp_vec_field = fs->cvelocity->get_cdata();
    tmp_vec_field->compute_stats(
            fs->kk,
            stat_group,
            "velocity",
            fs->iteration / niter_stat,
            max_velocity_estimate/sqrt(3));

    *tmp_vec_field = fs->cvorticity->get_cdata();
    tmp_vec_field->compute_stats(
            fs->kk,
            stat_group,
            "vorticity",
            fs->iteration / niter_stat,
            max_vorticity_estimate/sqrt(3));

    if (this->myrank == 0)
        H5Gclose(stat_group);

    if (myrank == 0)
    {
        std::string fname = (
                std::string("stop_") +
                std::string(simname));
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

template class NSVE<float>;
template class NSVE<double>;

