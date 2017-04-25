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

def compare_trajectories(
        c0, c1):
    """
        c0 is NSReader of NavierStokes data
        c1 is NSReader of NSVorticityEquation data
    """
    f = plt.figure(figsize = (6, 10))
    ntrajectories = 32

    a = f.add_subplot(211)
    pf = c0.get_particle_file()
    a.scatter(pf['tracers0/state'][0, :ntrajectories, 0],
              pf['tracers0/state'][0, :ntrajectories, 2])
    a.plot(pf['tracers0/state'][:, :ntrajectories, 0],
           pf['tracers0/state'][:, :ntrajectories, 2])
    a.set_xlabel('$x$')
    a.set_xlabel('$z$')
    c0_initial_condition = pf['tracers0/state'][0, :ntrajectories]
    pf.close()

    a = f.add_subplot(212)
    pf = h5py.File(c1.simname + '_checkpoint_0.h5', 'r')
    state = []
    nsteps = len(pf['tracers0/state'].keys())
    for ss in range(nsteps):
        state.append(pf['tracers0/state/{0}'.format(
            ss*c1.parameters['niter_out'])][:ntrajectories])
    state = np.array(state)
    c1_initial_condition = state[0, :]
    a.scatter(state[0, :, 0],
              state[0, :, 2])
    a.plot(state[:, :, 0],
           state[:, :, 2])
    a.set_xlabel('$x$')
    a.set_xlabel('$z$')
    f.tight_layout()
    f.savefig('figs/trajectories.pdf')

    print('difference between initial conditions is {0}'.format(
        np.max(np.abs(c0_initial_condition - c1_initial_condition))))
    return None

def main():
    niterations = 64
    c = bfps.NavierStokes(simname = 'fluid_solver')
    subprocess.call('rm *fluid_solver* NavierStokes*', shell = True)
    c.launch(
            ['-n', '32',
             '--simname', 'fluid_solver',
             '--ncpu', '4',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--nparticles', '100',
             '--particle-rand-seed', '2',
             '--niter_part', '1',
             '--wd', './'] +
            sys.argv[1:])
    subprocess.call('cat err_file_fluid_solver_0', shell = True)
    subprocess.call('rm *vorticity_equation* NSVE*', shell = True)
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
             '--checkpoints_per_file', '{0}'.format(2*niterations),
             '--nparticles', '100',
             '--particle-rand-seed', '2',
             '--wd', './'] +
            sys.argv[1:])
    subprocess.call('cat err_file_vorticity_equation_0', shell = True)
    c0 = NSReader(simname = 'fluid_solver')
    c1 = NSReader(simname = 'vorticity_equation')
    compare_moments(c0, c1)
    compare_trajectories(c0, c1)
    return None

if __name__ == '__main__':
    main()

