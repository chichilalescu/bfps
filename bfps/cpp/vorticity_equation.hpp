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

#include <sys/stat.h>
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
        int checkpoint;
        int checkpoints_per_file;

        /* fields */
        field<rnumber, be, THREE> *cvorticity, *cvelocity;
        field<rnumber, be, THREE> *rvorticity;
        kspace<be, SMOOTH> *kk;


        /* short names for velocity, and 4 vorticity fields */
        field<rnumber, be, THREE> *u, *v[4];

        /* physical parameters */
        double nu;
        int fmode;             // for Kolmogorov flow
        double famplitude;     // both for Kflow and band forcing
        double fk0, fk1;       // for band forcing
        double injection_rate; // for fixed energy injection rate
        double energy;         // for fixed energy
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

        /** \brief Method that computes force and adds it to the right hand side of the NS equations.
         *
         *   If the force has an explicit expression, as for instance in the case of Kolmogorov forcing,
         *   the term should be added to the nonlinear term for the purposes of time-stepping, since
         *   otherwise a custom time-stepping scheme would need to be implemented for each forcing type.
         *
         */
        void add_forcing(field<rnumber, be, THREE> *dst,
                         field<rnumber, be, THREE> *src_vorticity);

        /** \brief Method that imposes action of forcing on new vorticity field.
         *
         *   If the force is implicit, in the sense that kinetic energy must be
         *   preserved or something similar, then the action must be imposed
         *   after the non-linear term has been added.
         *
         */
        void impose_forcing(
                field<rnumber, be, THREE> *omega_new,
                field<rnumber, be, THREE> *omega_old);
        void compute_vorticity(void);
        void compute_velocity(field<rnumber, be, THREE> *vorticity);

        /* I/O stuff */
        inline std::string get_current_fname()
        {
            return (
                    std::string(this->name) +
                    std::string("_checkpoint_") +
                    std::to_string(this->checkpoint) +
                    std::string(".h5"));
        }
        void update_checkpoint(void);
        inline void io_checkpoint(bool read = true)
        {
            assert(!this->cvorticity->real_space_representation);
            if (!read)
                this->update_checkpoint();
            std::string fname = this->get_current_fname();
            this->cvorticity->io(
                    fname,
                    "vorticity",
                    this->iteration,
                    read);
            if (read)
            {
                #if (__GNUC__ <= 4 && __GNUC_MINOR__ < 7)
                    this->kk->low_pass<rnumber, THREE>(this->cvorticity->get_cdata(), this->kk->kM);
                    this->kk->force_divfree<rnumber>(this->cvorticity->get_cdata());
                #else
                    this->kk->template low_pass<rnumber, THREE>(this->cvorticity->get_cdata(), this->kk->kM);
                    this->kk->template force_divfree<rnumber>(this->cvorticity->get_cdata());
                #endif
            }
        }

        /* statistics and general postprocessing */
        void compute_pressure(field<rnumber, be, ONE> *pressure);
        void compute_Eulerian_acceleration(field<rnumber, be, THREE> *acceleration);
        void compute_Lagrangian_acceleration(field<rnumber, be, THREE> *acceleration);
};

#endif//VORTICITY_EQUATION

