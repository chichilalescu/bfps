#ifndef NSVE_NO_OUTPUT_HPP
#define NSVE_NO_OUTPUT_HPP

#include "full_code/NSVE.hpp"

template <typename rnumber>
class NSVE_no_output: public NSVE<rnumber>
{
    public:
    NSVE_no_output(
            const MPI_Comm COMMUNICATOR,
            const std::string &simulation_name):
        NSVE<rnumber>(
                COMMUNICATOR,
                simulation_name){}
    ~NSVE_no_output(){}
    int write_checkpoint(void)
    {
        return 0;
    }
    int read_parameters(void);
};

#endif//NSVE_NO_OUTPUT_HPP

