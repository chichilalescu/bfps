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



#ifndef NSVE_FIELD_STATS_HPP
#define NSVE_FIELD_STATS_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include "base.hpp"
#include "field.hpp"
#include "field_binary_IO.hpp"
#include "full_code/postprocess.hpp"

template <typename rnumber>
class NSVE_field_stats: public postprocess
{
    private:
        field_binary_IO<rnumber, COMPLEX, THREE> *bin_IO;
    public:
        field<rnumber, FFTW, THREE> *vorticity;

        NSVE_field_stats(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            postprocess(
                    COMMUNICATOR,
                    simulation_name){}
        virtual ~NSVE_field_stats(){}

        virtual int initialize(void);
        virtual int work_on_current_iteration(void);
        virtual int finalize(void);

        int read_current_cvorticity(void);
};

#endif//NSVE_FIELD_STATS_HPP

