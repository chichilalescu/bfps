/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "field.hpp"
#include "field_descriptor.hpp"

#ifndef VORTICITY_EQUATION

#define VORTICITY_EQUATION

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * This particular class is only meant as a stepping stone to a proper solver
 * that only uses the field class (and related layout and kspace classes), and
 * HDF5 for I/O.
 * */

template <typename rnumber,
          field_backend be>
class vorticity_equation
{
    public:
        /* name */
        char name[256];

        /* iteration */
        int iteration;

        /* fields */
        field<rnumber, be, THREE> *cvorticity, *cvelocity;
        field<rnumber, be, THREE> *rvorticity, *rvelocity;
        kspace<be, SMOOTH> *kk;


        /* short names for velocity, and 4 vorticity fields */
        field<rnumber, be, THREE> *u, *v[4];

        /* physical parameters */
        double nu;
        int fmode;         // for Kolmogorov flow
        double famplitude; // both for Kflow and band forcing
        double fk0, fk1;   // for band forcing
        char forcing_type[128];

        /* constructor, destructor */
        vorticity_equation(
                const char *NAME,
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0,
                unsigned FFTW_PLAN_RIGOR = FFTW_MEASURE);
        ~vorticity_equation(void);

        /* solver essential methods */
        void omega_nonlin(int src);
        void step(double dt);
        void impose_zero_modes(void);
        void add_forcing(field<rnumber, be, THREE> *dst,
                         field<rnumber, be, THREE> *src_vorticity,
                         rnumber factor);
        void compute_vorticity(void);
        void compute_velocity(field<rnumber, be, THREE> *vorticity);

        /* binary I/O stuff */
        inline void fill_up_filename(const char *base_name, char *full_name)
        {
            sprintf(full_name, "%s_%s_i%.5x.h5", this->name, base_name, this->iteration);
        }

        /* statistics and general postprocessing */
        void compute_pressure(field<rnumber, be, ONE> *pressure);
        void compute_Eulerian_acceleration(field<rnumber, be, THREE> *acceleration);
        void compute_Lagrangian_acceleration(field<rnumber, be, THREE> *acceleration);
};

#endif//VORTICITY_EQUATION

