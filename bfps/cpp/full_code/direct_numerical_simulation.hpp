/**********************************************************************
*                                                                     *
*  Copyright 2017 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/



#ifndef DIRECT_NUMERICAL_SIMULATION_HPP
#define DIRECT_NUMERICAL_SIMULATION_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include "base.hpp"
#include "full_code/code_base.hpp"

class direct_numerical_simulation: public code_base
{
    public:
        int checkpoint;
        int checkpoints_per_file;
        int niter_out;
        int niter_stat;
        int niter_todo;
        hid_t stat_file;

        direct_numerical_simulation(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            code_base(
                    COMMUNICATOR,
                    simulation_name){}
        virtual ~direct_numerical_simulation(){}

        virtual int write_checkpoint(void) = 0;
        virtual int initialize(void) = 0;
        virtual int step(void) = 0;
        virtual int do_stats(void) = 0;
        virtual int finalize(void) = 0;

        int main_loop(void);
        int read_iteration(void);
        int write_iteration(void);
        int grow_file_datasets(void);
};

#endif//DIRECT_NUMERICAL_SIMULATION_HPP

