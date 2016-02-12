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



#include <algorithm>
#include "particles_base.hpp"

template <int particle_type>
single_particle_state<particle_type>::single_particle_state()
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->data = new double[3];
            std::fill_n(this->data, 3, 0);
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type>::single_particle_state(
        const single_particle_state<particle_type> &src)
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            this->data = new double[3];
            std::copy(src.data, src.data + 3, this->data);
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type>::~single_particle_state()
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            delete[] this->data;
            break;
    }
}

template <int particle_type>
single_particle_state<particle_type> &single_particle_state<particle_type>::operator=(
        const single_particle_state &src)
{
    switch(particle_type)
    {
        case VELOCITY_TRACER:
            std::copy(src.data, src.data + 3, this->data);
            break;
    }
    return *this;
}



/*****************************************************************************/
template class single_particle_state<VELOCITY_TRACER>;
/*****************************************************************************/
