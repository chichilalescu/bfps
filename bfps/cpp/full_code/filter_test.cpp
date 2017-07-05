#include <string>
#include <cmath>
#include "filter_test.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int filter_test<rnumber>::initialize(void)
{
    this->read_parameters();
    this->scal_field = new field<rnumber, FFTW, ONE>(
            nx, ny, nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    this->kk = new kspace<FFTW, SMOOTH>(
            this->scal_field->clayout, this->dkx, this->dky, this->dkz);

    if (this->myrank == 0)
    {
        hid_t stat_file = H5Fopen(
                (this->simname + std::string(".h5")).c_str(),
                H5F_ACC_RDWR,
                H5P_DEFAULT);
        this->kk->store(stat_file);
        H5Fclose(stat_file);
    }
    this->read_iteration();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int filter_test<rnumber>::reset_field(
        int dimension)
{
    this->scal_field->real_space_representation = true;
    *this->scal_field = 0.0;
    if (this->scal_field->rlayout->starts[0] == 0)
    {
        switch(dimension)
        {
            case 0:
                this->scal_field->rval(0) = 1.0 / (
                        (4*acos(0) / (this->nx*this->dkx))*
                        (4*acos(0) / (this->ny*this->dky))*
                        (4*acos(0) / (this->nz*this->dkz)));
                break;
            case 1:
                for (ptrdiff_t xindex = 0; xindex < this->nx; xindex++)
                    this->scal_field->rval(this->scal_field->get_rindex(xindex, 0, 0)) = 1.0 / (
                        (4*acos(0) / (this->ny*this->dky))*
                        (4*acos(0) / (this->nz*this->dkz)));
                break;
            case 2:
                for (ptrdiff_t yindex = 0; yindex < this->ny; yindex++)
                for (ptrdiff_t xindex = 0; xindex < this->nx; xindex++)
                {
                    this->scal_field->rval(
                            this->scal_field->get_rindex(xindex, yindex, 0)) = 1.0 / (
                        (4*acos(0) / (this->nz*this->dkz)));
                }
                break;
            default:
                break;
        }
    }
    this->scal_field->dft();
    this->scal_field->symmetrize();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int filter_test<rnumber>::step(void)
{
    this->iteration = (this->iteration + this->niter_todo -
                       (this->iteration % this->niter_todo));
    return EXIT_SUCCESS;
}

template <typename rnumber>
int filter_test<rnumber>::write_checkpoint(void)
{
    std::string filename = this->simname + std::string("_fields.h5");
    for (int dimension = 0; dimension < 3; dimension++)
    {
        this->reset_field(dimension);
        this->kk->template filter<rnumber, ONE>(
                this->scal_field->get_cdata(),
                4*acos(0) / (this->filter_length),
                "sharp_Fourier_sphere");
        this->scal_field->ift();
        this->scal_field->normalize();
        this->scal_field->io(
                filename,
                "sharp_Fourier_sphere",
                dimension,
                false);
        this->reset_field(dimension);
        this->kk->template filter<rnumber, ONE>(
                this->scal_field->get_cdata(),
                4*acos(0) / (this->filter_length / sqrt(log(2))),
                "Gauss");
        this->scal_field->ift();
        this->scal_field->normalize();
        this->scal_field->io(
                filename,
                "Gauss",
                dimension,
                false);
        this->reset_field(dimension);
        this->kk->template filter<rnumber, ONE>(
                this->scal_field->get_cdata(),
                4*acos(0) / (this->filter_length*2),
                "ball");
        this->scal_field->ift();
        this->scal_field->normalize();
        this->scal_field->io(
                filename,
                "ball",
                dimension,
                false);
    }
    return EXIT_SUCCESS;
}

template <typename rnumber>
int filter_test<rnumber>::finalize(void)
{
    delete this->scal_field;
    delete this->kk;
    return EXIT_SUCCESS;
}

template <typename rnumber>
int filter_test<rnumber>::read_parameters()
{
    hid_t parameter_file;
    hid_t dset, memtype, space;
    char fname[256];
    char *string_data;
    sprintf(fname, "%s.h5", this->simname.c_str());
    parameter_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(parameter_file, "/parameters/niter_todo", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->niter_todo);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/niter_stat", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->niter_stat);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/niter_out", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->niter_out);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dealias_type", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dealias_type);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dkx", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dkx);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dky", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dky);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/dkz", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->dkz);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/nx", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nx);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/ny", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->ny);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/nz", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->nz);
    H5Dclose(dset);
    dset = H5Dopen(parameter_file, "/parameters/filter_length", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->filter_length);
    H5Dclose(dset);
    H5Fclose(parameter_file);
    return EXIT_SUCCESS;
}

template class filter_test<float>;
template class filter_test<double>;

