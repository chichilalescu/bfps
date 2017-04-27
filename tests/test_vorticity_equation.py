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



import sys
import os
import numpy as np
import h5py
import argparse
import subprocess

import bfps
import bfps.tools

from bfps_addons import NSReader
import matplotlib.pyplot as plt

def compare_moments(
        c0, c1):
    df0 = c0.get_data_file()
    df1 = c1.get_data_file()
    f = plt.figure(figsize=(6,10))
    a = f.add_subplot(211)
    a.plot(df0['statistics/moments/vorticity'][:, 2, 3],
           color = 'blue',
           marker = '+')
    a.plot(df1['statistics/moments/vorticity'][:, 2, 3],
           color = 'red',
           marker = 'x')
    a = f.add_subplot(212)
    a.plot(df0['statistics/moments/velocity'][:, 2, 3],
           color = 'blue',
           marker = '+')
    a.plot(df1['statistics/moments/velocity'][:, 2, 3],
           color = 'red',
           marker = 'x')
    f.tight_layout()
    f.savefig('figs/moments.pdf')
    return None

def overlap_trajectories(
        c0, c1):
    """
        c0 is NSReader of NavierStokes data
        c1 is NSReader of NSVorticityEquation data
    """
    f = plt.figure(figsize = (6, 6))
    ntrajectories = 100

    a = f.add_subplot(111)
    pf = c0.get_particle_file()
    a.scatter(pf['tracers0/state'][0, :ntrajectories, 0],
              pf['tracers0/state'][0, :ntrajectories, 2],
              marker = '+',
              color = 'blue')
    a.plot(pf['tracers0/state'][:, :ntrajectories, 0],
           pf['tracers0/state'][:, :ntrajectories, 2])
    a.set_xlabel('$x$')
    a.set_ylabel('$z$')
    c0_initial_condition = pf['tracers0/state'][0, :ntrajectories]
    pf.close()

    pf = h5py.File(c1.simname + '_checkpoint_0.h5', 'r')
    state = []
    nsteps = len(pf['tracers0/state'].keys())
    for ss in range(nsteps):
        state.append(pf['tracers0/state/{0}'.format(
            ss*c1.parameters['niter_out'])][:ntrajectories])
    state = np.array(state)
    c1_initial_condition = state[0, :]
    a.scatter(state[0, :, 0],
              state[0, :, 2],
              marker = 'x',
              color = 'red')
    a.plot(state[:, :, 0],
           state[:, :, 2],
           dashes = (1, 1))
    a.set_xlabel('$x$')
    a.set_ylabel('$z$')
    f.tight_layout()
    f.savefig('figs/trajectories.pdf')

    print('difference between initial conditions is {0}'.format(
        np.max(np.abs(c0_initial_condition - c1_initial_condition))))
    return None

def overlap_worst_trajectory(
        c0, c1):
    """
        c0 is NSReader of NavierStokes data
        c1 is NSReader of NSVorticityEquation data
    """

    ntrajectories = 100
    pf0 = c0.get_particle_file()
    state0 = pf0['tracers0/state'][:, :ntrajectories]
    pf0.close()

    pf1 = h5py.File(c1.simname + '_checkpoint_0.h5', 'r')
    state1 = []
    nsteps = len(pf1['tracers0/state'].keys())
    for ss in range(nsteps):
        state1.append(pf1['tracers0/state/{0}'.format(
            ss*c1.parameters['niter_out'])][:ntrajectories])
    state1 = np.array(state1)
    pf1.close()

    diff = np.abs(state0 - state1)
    bad_index = np.argmax(np.sum(diff[-1]**2, axis = 1))

    f = plt.figure(figsize = (6, 10))

    ax = f.add_subplot(311)
    ay = f.add_subplot(312)
    az = f.add_subplot(313)
    ax.set_ylabel('$x$')
    ax.set_xlabel('iteration')
    ay.set_ylabel('$y$')
    ay.set_xlabel('iteration')
    az.set_ylabel('$z$')
    az.set_xlabel('iteration')

    ax.plot(state0[:, bad_index, 0])
    ay.plot(state0[:, bad_index, 1])
    az.plot(state0[:, bad_index, 2])
    ax.plot(state1[:, bad_index, 0], dashes = (1, 1))
    ay.plot(state1[:, bad_index, 1], dashes = (1, 1))
    az.plot(state1[:, bad_index, 2], dashes = (1, 1))
    f.tight_layout()
    f.savefig('figs/trajectories.pdf')
    return None

def get_maximum_trajectory_error(
        c0, c1):
    """
        c0 is NSReader of NavierStokes data
        c1 is NSReader of NSVorticityEquation data
    """

    ntrajectories = 100
    pf0 = c0.get_particle_file()
    state0 = pf0['tracers0/state'][:, :ntrajectories]
    pf0.close()

    pf1 = h5py.File(c1.simname + '_checkpoint_0.h5', 'r')
    state1 = []
    nsteps = len(pf1['tracers0/state'].keys())
    for ss in range(nsteps):
        state1.append(pf1['tracers0/state/{0}'.format(
            ss*c1.parameters['niter_out'])][:ntrajectories])
    state1 = np.array(state1)
    pf1.close()

    diff = np.abs(state0 - state1)
    max_distance = np.max(diff, axis = 1)

    f = plt.figure(figsize = (6, 10))

    a = f.add_subplot(111)
    a.set_xlabel('iteration')

    a.plot(max_distance[:, 0], label = '$x$ difference')
    a.plot(max_distance[:, 1], label = '$y$ difference')
    a.plot(max_distance[:, 2], label = '$z$ difference')
    a.legend(loc = 'best')

    f.tight_layout()
    f.savefig('figs/trajectories.pdf')
    return None

def check_interpolation(
        c0, c1,
        nparticles = 2):
    """
        c0 is NSReader of NavierStokes data
        c1 is NSReader of NSVorticityEquation data
    """
    f = plt.figure(figsize = (6, 10))

    a = f.add_subplot(211)
    pf = c0.get_particle_file()
    x0 = pf['tracers0/state'][0, :, 0]
    y0 = pf['tracers0/state'][0, :, 2]
    v0 = np.sum(
            pf['tracers0/rhs'][1, 1]**2,
            axis = 1)**.5
    a.scatter(
            x0, y0,
            c = v0,
            vmin = v0.min(),
            vmax = v0.max(),
            edgecolors = 'none',
            s = 5.,
            cmap = plt.get_cmap('magma'))
    a.set_xlabel('$x$')
    a.set_ylabel('$z$')
    pf.close()

    a = f.add_subplot(212)
    pf = h5py.File(c1.simname + '_checkpoint_0.h5', 'r')
    state = pf['tracers0/state/0']
    x1 = state[:, 0]
    y1 = state[:, 2]
    v1 = np.sum(
            pf['tracers0/rhs/1'][1]**2,
            axis = 1)**.5
    # using v0 for colors on purpose, because we want the velocity to be the same,
    # so v1.min() should be equal to v0.min() etc.
    a.scatter(
            x1, y1,
            c = v1,
            vmin = v0.min(),
            vmax = v0.max(),
            edgecolors = 'none',
            s = 5.,
            cmap = plt.get_cmap('magma'))
    a.set_xlabel('$x$')
    a.set_ylabel('$z$')
    f.tight_layout()
    f.savefig('figs/trajectories.pdf')
    return None

def main():
    niterations = 64
    particle_initial_condition = None
    nparticles = 100
    run_NS = True
    run_NSVE = False
    plain_interpolation_test = False
    if plain_interpolation_test:
        niterations = 1
        pcloudX = np.pi
        pcloudY = np.pi
        particle_cloud_size = np.pi
        nparticles = 32*4
        particle_initial_condition = np.zeros(
                (nparticles,
                 nparticles,
                 3),
                dtype = np.float64)
        xvals = (pcloudX +
                 np.linspace(-particle_cloud_size/2,
                              particle_cloud_size/2,
                              nparticles))
        yvals = (pcloudY +
                 np.linspace(-particle_cloud_size/2,
                              particle_cloud_size/2,
                              nparticles))
        particle_initial_condition[..., 0] = xvals[None, None, :]
        particle_initial_condition[..., 2] = yvals[None, :, None]
        particle_initial_condition = particle_initial_condition.reshape(-1, 3)
        nparticles = nparticles**2
    c = bfps.NavierStokes(simname = 'fluid_solver')
    if run_NS:
        run_NSVE = True
        subprocess.call('rm *fluid_solver* NavierStokes*', shell = True)
        c.launch(
                ['-n', '32',
                 '--simname', 'fluid_solver',
                 '--ncpu', '4',
                 '--niter_todo', '{0}'.format(niterations),
                 '--niter_out', '{0}'.format(niterations),
                 '--niter_stat', '1',
                 '--nparticles', '{0}'.format(nparticles),
                 '--particle-rand-seed', '2',
                 '--niter_part', '1',
                 '--njobs', '2',
                 '--wd', './'] +
                sys.argv[1:],
                particle_initial_condition = particle_initial_condition)
        subprocess.call('cat err_file_fluid_solver_0', shell = True)
        subprocess.call('rm *vorticity_equation* NSVE*', shell = True)
    if run_NSVE:
        data = c.read_cfield(iteration = 0)
        f = h5py.File('vorticity_equation_checkpoint_0.h5', 'w')
        f['vorticity/complex/0'] = data
        f.close()
        c = bfps.NSVorticityEquation()
        c.launch(
                ['-n', '32',
                 '--simname', 'vorticity_equation',
                 '--np', '4',
                 '--ntpp', '1',
                 '--niter_todo', '{0}'.format(niterations),
                 '--niter_out', '1',
                 '--niter_stat', '1',
                 '--checkpoints_per_file', '{0}'.format(3*niterations),
                 '--nparticles', '{0}'.format(nparticles),
                 '--particle-rand-seed', '2',
                 '--njobs', '2',
                 '--wd', './'] +
                sys.argv[1:],
                particle_initial_condition = particle_initial_condition)
        subprocess.call('cat err_file_vorticity_equation_0', shell = True)
    c0 = NSReader(simname = 'fluid_solver')
    c1 = NSReader(simname = 'vorticity_equation')
    if plain_interpolation_test:
        check_interpolation(c0, c1, nparticles = int(nparticles**.5))
    else:
        get_maximum_trajectory_error(c0, c1)
        #overlap_worst_trajectory(c0, c1)
    return None

if __name__ == '__main__':
    main()

