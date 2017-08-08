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



#ifndef JOINT_ACC_VEL_STATS_HPP
#define JOINT_ACC_VEL_STATS_HPP

#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include "base.hpp"
#include "field.hpp"
#include "field_binary_IO.hpp"
#include "vorticity_equation.hpp"
#include "full_code/NSVE_field_stats.hpp"

template <typename rnumber>
class joint_acc_vel_stats: public NSVE_field_stats<rnumber>
{
    public:
        hid_t stat_file;
        kspace<FFTW, SMOOTH> *kk;
        vorticity_equation<rnumber, FFTW> *ve;

        int checkpoints_per_file;
        int niter_out;
        double max_acceleration_estimate;
        double max_velocity_estimate;

        joint_acc_vel_stats(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            NSVE_field_stats<rnumber>(
                    COMMUNICATOR,
                    simulation_name){}
        virtual ~joint_acc_vel_stats(){}

        int initialize(void);
        int work_on_current_iteration(void);
        int finalize(void);
};

#endif//JOINT_ACC_VEL_STATS_HPP

