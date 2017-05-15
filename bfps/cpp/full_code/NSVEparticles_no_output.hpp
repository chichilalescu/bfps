#ifndef NSVEPARTICLES_NO_OUTPUT_HPP
#define NSVEPARTICLES_NO_OUTPUT_HPP

#include "full_code/NSVEparticles.hpp"

template <typename rnumber>
class NSVEparticles_no_output: public NSVEparticles<rnumber>
{
    public:
    NSVEparticles_no_output(
            const MPI_Comm COMMUNICATOR,
            const std::string &simulation_name):
        NSVEparticles<rnumber>(
                COMMUNICATOR,
                simulation_name){}
    ~NSVEparticles_no_output(){}
    int write_checkpoint(void)
    {
        return 0;
    }
    int read_parameters(void);
};

#endif//NSVEPARTICLES_NO_OUTPUT_HPP

