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



#ifndef NSVE_HPP
#define NSVE_HPP



#include <cstdlib>
#include "base.hpp"
#include "vorticity_equation.hpp"
#include "full_code/direct_numerical_simulation.hpp"

template <typename rnumber>
class NSVE: public direct_numerical_simulation
{
    public:

        /* parameters that are read in read_parameters */
        double dt;
        double famplitude;
        double fk0;
        double fk1;
        int fmode;
        char forcing_type[512];
        int histogram_bins;
        double max_velocity_estimate;
        double max_vorticity_estimate;
        double nu;

        /* other stuff */
        vorticity_equation<rnumber, FFTW> *fs;
        field<rnumber, FFTW, THREE> *tmp_vec_field;
        field<rnumber, FFTW, ONE> *tmp_scal_field;


        NSVE(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            direct_numerical_simulation(
                    COMMUNICATOR,
                    simulation_name){}
        ~NSVE(){}

        int initialize(void);
        int main_loop(void);
        int finalize(void);

        int read_parameters(void);
        int do_stats(void);
};

#endif//NSVE_HPP

