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
import h5py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from test_base import *

def convergence_test(
        opt,
        code_launch_routine,
        init_vorticity = None,
        code_class = bfps.NavierStokes):
    opt.simname = 'N{0:0>4}_0'.format(opt.n)
    clist = []
    clist.append(code_launch_routine(
            opt,
            vorticity_field = init_vorticity,
            dt = 0.04,
            code_class = code_class))
    clist[0].compute_statistics()
    opt.initialize = True
    dtlist = []
    errlist = []
    for i in range(1, 5):
        dtlist.append(clist[-1].parameters['dt']*clist[-1].statistics['vel_max'] / (2*np.pi / clist[-1].parameters['nx']))
        opt.simname = 'N{0:0>4}_{1}'.format(opt.n, i)
        init_vorticity = np.fromfile(
                os.path.join(clist[0].work_dir, clist[0].simname + '_cvorticity_i00000'),
                dtype = clist[0].dtype)
        opt.niter_todo *= 2
        opt.niter_stat *= 2
        clist.append(code_launch_routine(
                opt,
                dt = clist[0].parameters['dt']/(2**i),
                vorticity_field = init_vorticity,
                code_class = code_class,
                tracer_state_file = h5py.File(os.path.join(clist[0].work_dir, clist[0].simname + '.h5'), 'r')))
        clist[-1].compute_statistics()
        f1 = np.fromfile(os.path.join(clist[-1].work_dir,
                                      clist[-1].simname + '_cvorticity_i{0:0>5x}'.format(clist[-1].parameters['niter_todo'])),
                         dtype = clist[-1].dtype)
        f2 = np.fromfile(os.path.join(clist[-2].work_dir,
                                      clist[-2].simname + '_cvorticity_i{0:0>5x}'.format(clist[-2].parameters['niter_todo'])),
                         dtype = clist[-2].dtype)
        errlist.append(np.max(np.abs(f1 - f2)) / np.max(f1))
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.plot(dtlist, errlist, marker = '.')
    a.plot(dtlist, np.array(dtlist), dashes = (1, 1))
    a.plot(dtlist, np.array(dtlist)**2, dashes = (2, 2))
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel('$\\|u\\|_\\infty \\frac{\\Delta t}{\\Delta x}$')
    fig.savefig('spec_err_vs_dt_{0}.pdf'.format(opt.precision))
    return None

if __name__ == '__main__':
    opt = parser.parse_args()
    convergence_test(opt, launch)

