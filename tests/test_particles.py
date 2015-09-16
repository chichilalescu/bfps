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
from test_convergence import convergence_test

# use ABC flow

def generate_ABC_flow(
        parameters = {'nx': 32,
                      'ny': 32,
                      'nz': 32},
        Fmode = 1,
        Famp = 1.0,
        dtype = np.complex64):
    Kdata = np.zeros((parameters['ny'],
                      parameters['nz'],
                      parameters['nx']//2+1,
                      3),
                     dtype = dtype)

    Kdata[                   Fmode, 0, 0, 0] =  Famp/2.
    Kdata[                   Fmode, 0, 0, 2] = -Famp/2.*1j
    Kdata[parameters['ny'] - Fmode, 0, 0, 0] =  Famp/2.
    Kdata[parameters['ny'] - Fmode, 0, 0, 2] =  Famp/2.*1j

    Kdata[0,                    Fmode, 0, 0] = -Famp/2.*1j
    Kdata[0,                    Fmode, 0, 1] =  Famp/2.
    Kdata[0, parameters['nz'] - Fmode, 0, 0] =  Famp/2.*1j
    Kdata[0, parameters['nz'] - Fmode, 0, 1] =  Famp/2.

    Kdata[0, 0,                    Fmode, 1] = -Famp/2.*1j
    Kdata[0, 0,                    Fmode, 2] =  Famp/2.
    return Kdata

def ABC_flow(x):
    return np.array([np.cos(x[1]) + np.sin(x[2]),
                     np.cos(x[2]) + np.sin(x[0]),
                     np.cos(x[0]) + np.sin(x[1])])

def Euler(x0, dt, nsteps, rhs):
    x = np.zeros((nsteps+1,) + x0.shape, x0.dtype)
    x[0] = x0
    for i in range(nsteps):
        x[i+1] = x[i] + dt*rhs(x[i])
    return x

def cRK(x0, dt, nsteps, rhs):
    x = np.zeros((nsteps+1,) + x0.shape, x0.dtype)
    x[0] = x0
    for i in range(nsteps):
        k1 = rhs(x[i])
        k2 = rhs(x[i] + dt*k1*0.5)
        k3 = rhs(x[i] + dt*k2*0.5)
        k4 = rhs(x[i] + dt*k3)
        x[i+1] = x[i] + dt*(k1 + 2*(k2 + k3) + k4)/6.
    return x

if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.precision == 'single':
        dtype = np.complex64
    elif opt.precision == 'double':
        dtype = np.complex128
    Kdata = generate_ABC_flow(
            parameters = {'nx': opt.n,
                          'ny': opt.n,
                          'nz': opt.n},
            dtype = dtype)
    c0, c1, c2 = convergence_test(
            opt,
            launch,
            code_class = FrozenFieldParticles,
            init_vorticity = Kdata)

    xE = Euler(c0.trajectories[1][0].T,
               c0.parameters['dt'],
               c0.parameters['niter_todo'],
               ABC_flow)
    xcRK = cRK(c0.trajectories[1][0].T,
               c0.parameters['dt'],
               c0.parameters['niter_todo'],
               ABC_flow)
    fig = plt.figure(figsize=(6,6))
    a = fig.add_subplot(111)
    for s in range(1, c0.particle_species):
        errlist = [np.average(np.abs(c.trajectories[s][-1, :, :3] - xcRK[-1].T))
                   for c in [c0, c1, c2]]
        dtlist = [c.parameters['dt']
                  for c in [c0, c1, c2]]
        a.plot(dtlist,
               errlist,
               label = '{0}'.format(s),
               marker = '.')
        a.plot(dtlist,
               np.array(dtlist)**s,
               label = '$\\Delta t^{0}$'.format(s),
               dashes = (2, s),
               color = (0, 0, 0))
    a.set_xscale('log')
    a.set_yscale('log')
    a.legend(loc = 'best')
    fig.savefig('test_particles_err_vs_cRK.pdf')

