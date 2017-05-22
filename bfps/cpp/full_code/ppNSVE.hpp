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



#ifndef PPNSVE_HPP
#define PPNSVE_HPP



#include <cstdlib>
#include "base.hpp"
#include "vorticity_equation.hpp"
#include "full_code/direct_numerical_simulation.hpp"
#include "full_code/NSVE.hpp"

template <typename rnumber>
class ppNSVE: public NSVE<rnumber>
{
    public:


        ppNSVE(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            NSVE<rnumber>(
                    COMMUNICATOR,
                    simulation_name){}
        ~ppNSVE(){}

        int initialize(void);
        int step(void);
        int finalize(void);

        virtual int read_parameters(void);
        int write_checkpoint(void);
        int do_stats(void);
        int main_loop(void);
};

#endif//PPNSVE_HPP

