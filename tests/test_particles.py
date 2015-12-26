#! /usr/bin/env python2
#######################################################################
#                                                                     #
#  Copyright 2015 Max Planck Institute                                #
#                 for Dynamics and Self-Organization                  #
#                                                                     #
#  This file is part of bfps.                                         #
#                                                                     #
#  bfps is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published  #
#  by the Free Software Foundation, either version 3 of the License,  #
#  or (at your option) any later version.                             #
#                                                                     #
#  bfps is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of     #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
#  GNU General Public License for more details.                       #
#                                                                     #
#  You should have received a copy of the GNU General Public License  #
#  along with bfps.  If not, see <http://www.gnu.org/licenses/>       #
#                                                                     #
# Contact: Cristian.Lalescu@ds.mpg.de                                 #
#                                                                     #
#######################################################################



import numpy as np
import matplotlib.pyplot as plt

from base import *

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

def AdamsBashforth(x0, dt, nsteps, rhs, nsubsteps = 1):
    x = np.zeros((nsteps+1,) + x0.shape, x0.dtype)
    x[0] = x0
    R = np.zeros((nsubsteps,) + x0.shape, x0.dtype)
    for i in range(nsteps):
        R[0] = rhs(x[i])
        if min(i, nsubsteps) == 2:
            x[i+1] = x[i] + dt*(3*R[0] - R[1])/2
        elif min(i, nsubsteps) == 3:
            x[i+1] = x[i] + dt*(23*R[0] - 16*R[1] + 5*R[2])/12
        elif min(i, nsubsteps) == 4:
            x[i+1] = x[i] + dt*(55*R[0] - 59*R[1] + 37*R[2] - 9*R[3])/24
        else:
            x[i+1] = x[i] + dt*R[0]
        # roll rhs:
        for j in range(nsubsteps-2, -1, -1):
            R[j+1] = R[j]
    return x

class err_finder:
    def __init__(self, clist):
        self.clist = clist
        self.xcRK = []
        for c in self.clist:
            self.xcRK.append(cRK(
                c.trajectories[1][0].T,
                c.parameters['dt'],
                c.parameters['niter_todo'],
                ABC_flow))
        self.dtlist = [c.parameters['dt']
                       for c in self.clist]
        self.ctraj = [None]
        for i in range(1, self.clist[0].particle_species):
            self.ctraj.append([self.clist[j].trajectories[i].transpose((0, 2, 1))
                               for j in range(len(self.clist))])
        return None
    def get_AB_err(self, nsubsteps = 1):
        self.xAB = []
        for c in self.clist:
            self.xAB.append(AdamsBashforth(
                c.trajectories[1][0].T,
                c.parameters['dt'],
                c.parameters['niter_todo'],
                ABC_flow,
                nsubsteps = 3))
        errlist = [np.average(np.abs(self.xAB[i][-1].T - self.xcRK[i][-1].T))
                   for i in range(len(self.clist))]
        return errlist
    def get_spec_error(self, traj_list):
        self.xx = []
        errlist = []
        for i in range(len(self.clist)):
            self.xx.append(cRK(
                traj_list[i][self.clist[i].parameters['niter_todo']//2],
                self.clist[i].parameters['dt'],
                self.clist[i].parameters['niter_todo']//2,
                ABC_flow))
        errlist = [np.average(np.abs(traj_list[i][-1].T - self.xx[i][-1].T))
                   for i in range(len(self.clist))]
        return errlist

if __name__ == '__main__':
    opt = parser.parse_args(
            ['-n', '16',
             '--run',
             '--initialize',
             '--frozen',
             '--ncpu', '2',
             '--nparticles', '128',
             '--niter_todo', '16',
             '--precision', 'single',
             '--wd', 'data/single'] +
            sys.argv[1:])
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

    ef = err_finder([c0, c1, c2])
    fig = plt.figure(figsize=(6,6))
    a = fig.add_subplot(111)
    for s in range(1, 5):
        ef.get_AB_err(s)
        errlist = [np.average(np.abs(ef.clist[i].trajectories[s][-1, :, :3] - ef.xAB[i][-1].T))
                   for i in range(len(ef.clist))]
        a.plot(ef.dtlist, errlist,
               label = 'directAB{0}'.format(s),
               marker = '.')
        a.plot(ef.dtlist, ef.get_spec_error(ef.ctraj[s]),
               label = 'specAB{0}'.format(s),
               marker = '.',
               dashes = (s, s))
        a.plot(ef.dtlist,
               np.array(ef.dtlist)**s,
               label = '$\\Delta t^{0}$'.format(s),
               dashes = (2, s),
               color = (0, 0, 0))
    a.set_xscale('log')
    a.set_yscale('log')
    a.legend(loc = 'lower right')
    a.set_xlim(.9*min(ef.dtlist), 2*max(ef.dtlist))
    fig.savefig('test_particles_evdt_' + opt.precision + '.pdf')
    fig = plt.figure(figsize=(6,6))
    a = fig.add_subplot(111)
    for t in range(c0.parameters['nparticles']):
        a.plot(ef.xcRK[0][:, 0, t],
               ef.xcRK[0][:, 1, t])
        a.plot(ef.xx[0][:, 0, t],
               ef.xx[0][:, 1, t],
               dashes = (1,1))
    fig.savefig('test_particles_traj_' + opt.precision + '.pdf')

