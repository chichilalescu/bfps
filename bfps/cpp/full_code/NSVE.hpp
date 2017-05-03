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

int grow_single_dataset(hid_t dset, int tincrement);

herr_t grow_dataset_visitor(
    hid_t o_id,
    const char *name,
    const H5O_info_t *info,
    void *op_data);

template <typename rnumber>
class NSVE: public direct_numerical_simulation
{
    public:

        /* parameters that are read in read_parameters */
        int checkpoints_per_file;
        int dealias_type;
        double dkx;
        double dky;
        double dkz;
        double dt;
        double famplitude;
        double fk0;
        double fk1;
        int fmode;
        char forcing_type[512];
        int histogram_bins;
        double max_velocity_estimate;
        double max_vorticity_estimate;
        int niter_out;
        int niter_part;
        int niter_stat;
        int niter_todo;
        int nparticles;
        double nu;
        int nx;
        int ny;
        int nz;

        /* other stuff */
        int iteration, checkpoint;
        hid_t stat_file;
        bool stop_code_now;
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

        int read_iteration(void);
        int write_iteration(void);
        int read_parameters(void);
        int grow_file_datasets(void);
        int do_stats(void);
};

#endif//NSVE_HPP

