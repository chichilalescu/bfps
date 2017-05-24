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

template <typename rnumber>
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
        this->print_simple_timer();
    }
    this->do_stats();
    this->print_simple_timer();
    if (this->iteration % this->niter_out != 0)
        this->write_checkpoint();
    return EXIT_SUCCESS;
}

template class NSVEparticles<float>;
template class NSVEparticles<double>;

