/***********************************************************************
*
*  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: Cristian.Lalescu@ds.mpg.de
*
************************************************************************/



#include "slab_field_particles.hpp"

#ifndef TRACERS

#define TRACERS

extern int myrank, nprocs;

template <class rnumber>
class tracers:public slab_field_particles<rnumber>
{
    public:
        rnumber *source_data;
        rnumber *data;

        /* methods */
        tracers(
                const char *NAME,
                fluid_solver_base<rnumber> *FSOLVER,
                const int NPARTICLES,
                const int BUFFERSIZE,
                rnumber *SOURCE_DATA);
        ~tracers();

        void transfer_data(bool clip_on = true);
        virtual void get_rhs(double *x, double *rhs);
        virtual void jump_estimate(double *jump_length);
};


#endif//TRACERS

