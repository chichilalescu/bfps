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



#include "interpolator_base.hpp"

#ifndef PARTICLES_BASE

#define PARTICLES_BASE

/* particle types */
enum particle_types {VELOCITY_TRACER};

/* 1 particle state type */

template <int particle_type>
class single_particle_state
{
    public:
        double *data;

        single_particle_state();
        single_particle_state(const single_particle_state &src);
        single_particle_state(const double *src);
        ~single_particle_state();

        single_particle_state<particle_type> &operator=(const single_particle_state &src);
        single_particle_state<particle_type> &operator=(const double *src);
};

#endif//PARTICLES_BASE

