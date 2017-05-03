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


#include "base.hpp"

class direct_numerical_simulation
{
    public:
        int myrank, nprocs;
        MPI_Comm comm;

        std::string simname;

        direct_numerical_simulation(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name);
        ~direct_numerical_simulation(){}

        virtual int initialize(void) = 0;
        virtual int main_loop(void) = 0;
        virtual int finalize(void) = 0;
};

#endif//DIRECT_NUMERICAL_SIMULATION_HPP

