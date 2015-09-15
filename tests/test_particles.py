#! /usr/bin/env python2
########################################################################
#
#  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: Cristian.Lalescu@ds.mpg.de
#
########################################################################

from test_base import *

from test_frozen_field import FrozenFieldParticles

# use ABC flow

def generate_ABC_flow(c, Fmode, Famp):
    Kdata = np.zeros((c.parameters['ny'],
                      c.parameters['nz'],
                      c.parameters['nx']//2+1,
                      3),
                     dtype = c.ctype)

    Kdata[                     Fmode, 0, 0, 0] =  Famp/2.
    Kdata[                     Fmode, 0, 0, 2] = -Famp/2.*1j
    Kdata[c.parameters['ny'] - Fmode, 0, 0, 0] =  Famp/2.
    Kdata[c.parameters['ny'] - Fmode, 0, 0, 2] =  Famp/2.*1j

    Kdata[0,                      Fmode, 0, 0] = -Famp/2.*1j
    Kdata[0,                      Fmode, 0, 1] =  Famp/2.
    Kdata[0, c.parameters['nz'] - Fmode, 0, 0] =  Famp/2.*1j
    Kdata[0, c.parameters['nz'] - Fmode, 0, 1] =  Famp/2.

    Kdata[0, 0,                      Fmode, 1] = -Famp/2.*1j
    Kdata[0, 0,                      Fmode, 2] =  Famp/2.
    return Kdata

def FFPlaunch(
        opt,
        nu = None,
        tracer_state_file = None):
    c = FrozenFieldParticles(
            work_dir = opt.work_dir,
            fluid_precision = opt.precision)
    assert((opt.nsteps % 4) == 0)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    if type(nu) == type(None):
        c.parameters['nu'] = 5.5*opt.n**(-4./3)
    else:
        c.parameters['nu'] = nu
    c.parameters['dt'] = 1e-2 * (64. / opt.n)
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['niter_out'] = opt.nsteps
    c.parameters['niter_part'] = 1
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = opt.nparticles
    c.add_particles(kcut = 'fs->kM/2')
    c.add_particles(integration_steps = 1, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.add_particles(integration_steps = 2, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.add_particles(integration_steps = 3, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.add_particles(integration_steps = 4, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.add_particles(integration_steps = 5, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.add_particles(integration_steps = 6, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.fill_up_fluid_code()
    c.finalize_code()
    c.write_src()
    c.write_par()
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            Kdata = generate_ABC_flow(c, 1, 1)
            Kdata.tofile(
                    os.path.join(c.work_dir,
                                 c.simname + "_c{0}_i{1:0>5x}".format('vorticity', opt.iteration)))
        if opt.iteration == 0:
            for species in range(c.particle_species):
                if type(tracer_state_file) == type(None):
                    data = None
                else:
                    data = tracer_state_file['particles/tracers{0}/state'.format(species)][0]
                c.generate_tracer_state(
                        species = species,
                        write_to_file = False,
                        testing = True,
                        rseed = 3284,
                        data = data)
        c.run(ncpu = opt.ncpu,
              njobs = opt.njobs)
    return c

from test_convergence import convergence_test

if __name__ == '__main__':
    convergence_test(parser.parse_args(), FFPlaunch)

