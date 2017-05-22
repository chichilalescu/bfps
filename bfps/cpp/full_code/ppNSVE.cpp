#include <string>
#include <cmath>
#include "NSVEparticles.hpp"
#include "scope_timer.hpp"


template <typename rnumber>
int NSVEparticles<rnumber>::initialize(void)
{
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::step(void)
{
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::write_checkpoint(void)
{
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::finalize(void)
{
    this->NSVE<rnumber>::finalize();
    return EXIT_SUCCESS;
}

template <typename rnumber>
int NSVEparticles<rnumber>::do_stats()
{
    return EXIT_SUCCESS;
}

template class NSVEparticles<float>;
template class NSVEparticles<double>;

