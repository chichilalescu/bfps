#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "scope_timer.hpp"
#include "hdf5_tools.hpp"
#include "full_code/postprocess.hpp"


int postprocess::main_loop(void)
{
    this->start_simple_timer();
    for (unsigned int iteration_counter = 0;
         iteration_counter < iteration_list.size();
         iteration_counter++)
    {
        this->iteration = iteration_list[iteration_counter];
    #ifdef USE_TIMINGOUTPUT
        const std::string loopLabel = ("postprocess::main_loop-" +
                                       std::to_string(this->iteration));
        TIMEZONE(loopLabel.c_str());
    #endif
        this->work_on_current_iteration();
        this->print_simple_timer(
                "iteration " + std::to_string(this->iteration));
    }
    return EXIT_SUCCESS;
}

