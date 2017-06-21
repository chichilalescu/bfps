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



#ifndef NSVEPARTICLES_HPP
#define NSVEPARTICLES_HPP



#include <cstdlib>
#include "base.hpp"
#include "vorticity_equation.hpp"
#include "full_code/NSVE.hpp"
#include "particles/particles_system_builder.hpp"
#include "particles/particles_output_hdf5.hpp"

/** \brief Navier-Stokes solver that includes simple Lagrangian tracers.
 *
 *  Child of Navier Stokes vorticity equation solver, this class calls all the
 *  methods from `NSVE`, and in addition integrates simple Lagrangian tracers
 *  in the resulting velocity field.
 */

template <typename rnumber>
class NSVEparticles: public NSVE<rnumber>
{
    public:

        /* parameters that are read in read_parameters */
        int niter_part;
        int nparticles;
        int tracers0_integration_steps;
        int tracers0_neighbours;
        int tracers0_smoothness;

        /* other stuff */
        std::unique_ptr<abstract_particles_system<long long int, double>> ps;
        particles_output_hdf5<long long int, double,3,3> *particles_output_writer_mpi;


        NSVEparticles(
                const MPI_Comm COMMUNICATOR,
                const std::string &simulation_name):
            NSVE<rnumber>(
                    COMMUNICATOR,
                    simulation_name){}
        ~NSVEparticles(){}

        int initialize(void);
        int step(void);
        int finalize(void);

        int read_parameters(void);
        int write_checkpoint(void);
        int do_stats(void);
};

#endif//NSVEPARTICLES_HPP

