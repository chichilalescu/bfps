#include <string>
#include <cmath>
#include "native_binary_to_hdf5.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int native_binary_to_hdf5<rnumber>::initialize(void)
{
    this->read_parameters();
    this->vec_field = new field<rnumber, FFTW, THREE>(
            nx, ny, nz,
            this->comm,
            DEFAULT_FFTW_FLAG);
    return EXIT_SUCCESS;
}

template <typename rnumber>
int native_binary_to_hdf5<rnumber>::work_on_current_iteration(void)
{
    return EXIT_SUCCESS;
}

template <typename rnumber>
int native_binary_to_hdf5<rnumber>::finalize(void)
{
    delete this->vec_field;
    return EXIT_SUCCESS;
}

template class native_binary_to_hdf5<float>;
template class native_binary_to_hdf5<double>;

