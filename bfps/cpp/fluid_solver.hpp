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
#include "field_descriptor.hpp"
#include "fluid_solver_base.hpp"

#ifndef FLUID_SOLVER

#define FLUID_SOLVER

extern int myrank, nprocs;


/* container for field descriptor, fields themselves, parameters, etc
 * using the same big macro idea that they're using in fftw3.h
 * I feel like I should quote:  Ugh.
 * */

template <class rnumber>
class fluid_solver:public fluid_solver_base<rnumber>
{
    public:
        /* fields */
        rnumber *rvorticity;
        rnumber *rvelocity ;
        typename fluid_solver_base<rnumber>::cnumber *cvorticity;
        typename fluid_solver_base<rnumber>::cnumber *cvelocity ;

        /* short names for velocity, and 4 vorticity fields */
        rnumber *ru, *rv[4];
        typename fluid_solver_base<rnumber>::cnumber *cu, *cv[4];

        /* plans */
        void *c2r_vorticity;
        void *r2c_vorticity;
        void *c2r_velocity;
        void *r2c_velocity;
        void *uc2r, *ur2c;
        void *vr2c[3], *vc2r[3];

        /* physical parameters */
        double nu;
        int fmode;         // for Kolmogorov flow
        double famplitude; // both for Kflow and band forcing
        double fk0, fk1;   // for band forcing
        char forcing_type[128];

        /* methods */
        fluid_solver(
                const char *NAME,
                int nx,
                int ny,
                int nz,
                double DKX = 1.0,
                double DKY = 1.0,
                double DKZ = 1.0,
                int DEALIAS_TYPE = 1,
                unsigned FFTW_PLAN_RIGOR = FFTW_MEASURE);
        ~fluid_solver(void);

        void compute_gradient_statistics(
                rnumber (*__restrict__ vec)[2],
                double *__restrict__ gradu_moments,
                double *__restrict__ trS2_Q_R_moments,
                ptrdiff_t *__restrict__ gradu_histograms,
                ptrdiff_t *__restrict__ trS2_Q_R_histograms,
                ptrdiff_t *__restrict__ QR2D_histogram,
                double trS2_Q_R_max_estimates[3],
                double gradu_max_estimates[9],
                const int nbins_1D = 256,
                const int nbins_2D = 64);

        void compute_vorticity(void);
        void compute_velocity(rnumber (*__restrict__ vorticity)[2]);
        void compute_pressure(rnumber (*__restrict__ pressure)[2]);
        void compute_Eulerian_acceleration(rnumber *__restrict__ dst);
        void compute_Lagrangian_acceleration(rnumber *__restrict__ dst);
        void ift_velocity();
        void dft_velocity();
        void ift_vorticity();
        void dft_vorticity();
        void omega_nonlin(int src);
        void step(double dt);
        void impose_zero_modes(void);
        void add_forcing(rnumber (*__restrict__ acc_field)[2], rnumber (*__restrict__ vort_field)[2], rnumber factor);

        int read(char field, char representation);
        int write(char field, char representation);
        int write_rTrS2();
        int write_renstrophy();
};

#endif//FLUID_SOLVER

