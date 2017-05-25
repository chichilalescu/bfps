#include <string>
#include <cmath>
#include "get_rfields.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int get_rfields<rnumber>::initialize(void)
{
    this->NSVE_field_stats<rnumber>::initialize();
    this->kk = new kspace<FFTW, SMOOTH>(
            this->vorticity->clayout, this->dkx, this->dky, this->dkz);
    hid_t parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    hid_t dset = H5Dopen(parameter_file, "/parameters/niter_out", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->niter_out);
    H5Dclose(dset);
    if (H5Lexists(parameter_file, "/parameters/checkpoints_per_file", H5P_DEFAULT))
    {
        dset = H5Dopen(parameter_file, "/parameters/checkpoints_per_file", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->checkpoints_per_file);
        H5Dclose(dset);
    }
    else
        this->checkpoints_per_file = 1;
    this->iteration_list = hdf5_tools::read_vector<int>(
            parameter_file,
            "/get_rfields/iteration_list");
    H5Fclose(parameter_file);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int get_rfields<rnumber>::work_on_current_iteration(void)
{
    DEBUG_MSG("entered get_rfields::work_on_current_iteration\n");
    this->read_current_cvorticity();
    field<rnumber, FFTW, THREE> *vel = new field<rnumber, FFTW, THREE>(
            this->nx, this->ny, this->nz,
            this->comm,
            this->vorticity->fftw_plan_rigor);

    vel->real_space_representation = false;
    this->kk->CLOOP_K2(
                [&](ptrdiff_t cindex,
                    ptrdiff_t xindex,
                    ptrdiff_t yindex,
                    ptrdiff_t zindex,
                    double k2){
        if (k2 <= this->kk->kM2 && k2 > 0)
        {
            vel->cval(cindex,0,0) = -(this->kk->ky[yindex]*this->vorticity->cval(cindex,2,1) - this->kk->kz[zindex]*this->vorticity->cval(cindex,1,1)) / k2;
            vel->cval(cindex,0,1) =  (this->kk->ky[yindex]*this->vorticity->cval(cindex,2,0) - this->kk->kz[zindex]*this->vorticity->cval(cindex,1,0)) / k2;
            vel->cval(cindex,1,0) = -(this->kk->kz[zindex]*this->vorticity->cval(cindex,0,1) - this->kk->kx[xindex]*this->vorticity->cval(cindex,2,1)) / k2;
            vel->cval(cindex,1,1) =  (this->kk->kz[zindex]*this->vorticity->cval(cindex,0,0) - this->kk->kx[xindex]*this->vorticity->cval(cindex,2,0)) / k2;
            vel->cval(cindex,2,0) = -(this->kk->kx[xindex]*this->vorticity->cval(cindex,1,1) - this->kk->ky[yindex]*this->vorticity->cval(cindex,0,1)) / k2;
            vel->cval(cindex,2,1) =  (this->kk->kx[xindex]*this->vorticity->cval(cindex,1,0) - this->kk->ky[yindex]*this->vorticity->cval(cindex,0,0)) / k2;
        }
        else
            std::fill_n((rnumber*)(vel->get_cdata()+3*cindex), 6, 0.0);
    }
    );
    vel->symmetrize();
    vel->ift();

    std::string fname = (
            this->simname +
            std::string("_checkpoint_") +
            std::to_string(this->iteration / (this->niter_out*this->checkpoints_per_file)) +
            std::string(".h5"));
    vel->io(
            fname,
            "velocity",
            this->iteration,
            false);

    delete vel;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int get_rfields<rnumber>::finalize(void)
{
    delete this->kk;
    this->NSVE_field_stats<rnumber>::finalize();
    return EXIT_SUCCESS;
}

template class get_rfields<float>;
template class get_rfields<double>;

