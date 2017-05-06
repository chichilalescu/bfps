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



#ifndef NSVEP_HPP
#define NSVEP_HPP



#include <cstdlib>
#include "base.hpp"
#include "vorticity_equation.hpp"
#include "full_code/direct_numerical_simulation.hpp"
#include "particles/particles_system_builder.hpp"
#include "particles/particles_output_hdf5.hpp"

template <typename rnumber>
class NSVEp: public direct_numerical_simulation
{
    public:

        /* parameters that are read in read_parameters */
        double dt;
        double famplitude;
        double fk0;
        double fk1;
        double nu;
        int fmode;
        char forcing_type[512];
        int histogram_bins;
        double max_velocity_estimate;
        double max_vorticity_estimate;

        int niter_part;
        int nparticles;
        int tracers0_integration_steps;
        int tracers0_neighbours;
        int tracers0_smoothness;

        /* other stuff */
        vorticity_equation<rnumber, FFTW> *fs;
        field<rnumber, FFTW, THREE> *tmp_vec_field;
        field<rnumber, FFTW, ONE> *tmp_scal_field;
        std::unique_ptr<abstract_particles_system<long long int, double>> ps;
        particles_output_hdf5<long long int, double,3,3> *particles_output_writer_mpi;


        NSVEp(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            direct_numerical_simulation(
                    COMMUNICATOR,
                    simulation_name){}
        ~NSVEp(){}

        int initialize(void);
        int step(void);
        int finalize(void);

        int read_parameters(void);
        int write_checkpoint(void);
        int do_stats(void);
};

#endif//NSVEP_HPP

