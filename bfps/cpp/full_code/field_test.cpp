#include <string>
#include <cmath>
#include <random>
#include "field_test.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int field_test<rnumber>::initialize(void)
{
    this->read_parameters();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int field_test<rnumber>::finalize(void)
{
    return EXIT_SUCCESS;
}

template <typename rnumber>
int field_test<rnumber>::read_parameters()
{
    this->test::read_parameters();
    // in case any parameters are needed, this is where they should be read
    hid_t parameter_file;
    hid_t dset;
    parameter_file = H5Fopen(
            (this->simname + std::string(".h5")).c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT);
    dset = H5Dopen(parameter_file, "/parameters/filter_length", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &this->filter_length);
    H5Dclose(dset);
    H5Fclose(parameter_file);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int field_test<rnumber>::do_work(void)
{
    // allocate
    field<rnumber, FFTW, ONE> *scal_field = new field<rnumber, FFTW, ONE>(
            this->nx, this->ny, this->nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    field<rnumber, FFTW, ONE> *scal_field_alt = new field<rnumber, FFTW, ONE>(
            this->nx, this->ny, this->nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    std::default_random_engine rgen;
    std::normal_distribution<rnumber> rdist;
    rgen.seed(1);
    //auto gaussian = std::bind(rgen, rdist);
    kspace<FFTW,SMOOTH> *kk = new kspace<FFTW, SMOOTH>(
            scal_field->clayout, this->dkx, this->dky, this->dkz);

    if (this->myrank == 0)
    {
        hid_t stat_file = H5Fopen(
                (this->simname + std::string(".h5")).c_str(),
                H5F_ACC_RDWR,
                H5P_DEFAULT);
        kk->store(stat_file);
        H5Fclose(stat_file);
    }

    // fill up scal_field
    scal_field->real_space_representation = true;
    scal_field->RLOOP(
            [&](ptrdiff_t rindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex){
            scal_field->rval(rindex) = rdist(rgen);
            });

    *scal_field_alt = scal_field->get_rdata();
    scal_field->dft();
    scal_field->ift();
    scal_field->normalize();

    double max_error = 0;
    scal_field->RLOOP(
            [&](ptrdiff_t rindex,
                ptrdiff_t xindex,
                ptrdiff_t yindex,
                ptrdiff_t zindex){
            double tval = fabs(scal_field->rval(rindex) - scal_field_alt->rval(rindex));
            if (max_error < tval)
                max_error = tval;
            });

    DEBUG_MSG("maximum error is %g\n", max_error);

    // deallocate
    delete kk;
    delete scal_field;
    delete scal_field_alt;
    return EXIT_SUCCESS;
}

template class field_test<float>;
template class field_test<double>;

