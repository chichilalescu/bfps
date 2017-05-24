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



#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include "base.hpp"
#include "full_code/code_base.hpp"

class postprocess: public code_base
{
    public:
        std::vector<int> iteration_list;
        hid_t stat_file;

        postprocess(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            code_base(
                    COMMUNICATOR,
                    simulation_name){}
        virtual ~postprocess(){}

        virtual int initialize(void) = 0;
        virtual int work_on_current_iteration(void) = 0;
        virtual int finalize(void) = 0;

        int main_loop(void);
};

#endif//POSTPROCESS_HPP

