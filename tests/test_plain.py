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



from test_base import *

import numpy as np
import matplotlib.pyplot as plt

parser.add_argument('--multiplejob',
        dest = 'multiplejob', action = 'store_true')

def plain(opt):
    wd = opt.work_dir
    opt.work_dir = wd + '/N{0:0>3x}_1'.format(opt.n)
    c0 = launch(opt, dt = 0.2/opt.n)
    c0.compute_statistics()
    df = c0.get_data_file()
    for s in range(c0.particle_species):
        acceleration_test(c0, species = s, m = 1)
    if not opt.multiplejob:
        return None
    assert(opt.niter_todo % 3 == 0)
    opt.work_dir = wd + '/N{0:0>3x}_2'.format(opt.n)
    opt.njobs *= 2
    opt.niter_todo /= 2
    c1 = launch(opt)
    c1.compute_statistics()
    opt.work_dir = wd + '/N{0:0>3x}_3'.format(opt.n)
    opt.njobs = 3*opt.njobs/2
    opt.niter_todo = 2*opt.niter_todo/3
    c2 = launch(opt)
    c2.compute_statistics()
    # plot energy and enstrophy
    fig = plt.figure(figsize = (12, 12))
    a = fig.add_subplot(221)
    c0.set_plt_style({'label' : '1',
                      'dashes' : (None, None),
                      'color' : (1, 0, 0)})
    c1.set_plt_style({'label' : '2',
                      'dashes' : (2, 2),
                      'color' : (0, 0, 1)})
    c2.set_plt_style({'label' : '3',
                      'dashes' : (3, 3),
                      'color' : (0, 1, 0)})
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['energy(t)'],
               label = c.style['label'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('energy')
    a.legend(loc = 'best')
    a = fig.add_subplot(222)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['enstrophy(t)'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('enstrophy')
    a = fig.add_subplot(223)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['kM']*c.statistics['etaK(t)'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('$k_M \\eta_K$')
    a = fig.add_subplot(224)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['vel_max(t)'] * (c.parameters['dt'] * c.parameters['dkx'] /
                                             (2*np.pi / c.parameters['nx'])),
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('$\\frac{\\Delta t \\| u \\|_\infty}{\\Delta x}$')
    fig.savefig('plain_stats_{0}.pdf'.format(opt.precision), format = 'pdf')

    fig = plt.figure(figsize = (12, 12))
    a = fig.add_subplot(221)
    a.plot(c0.statistics['t'],
           c0.statistics['energy(t)'] - c1.statistics['energy(t)'])
    a.set_title('energy')
    a = fig.add_subplot(222)
    a.plot(c0.statistics['t'],
           c0.statistics['enstrophy(t)'] - c1.statistics['enstrophy(t)'])
    a.set_title('enstrophy')
    a = fig.add_subplot(223)
    a.plot(c0.statistics['t'],
           c0.statistics['kM']*c0.statistics['etaK(t)'] - c1.statistics['kM']*c1.statistics['etaK(t)'])
    a.set_title('$k_M \\eta_K$')
    a = fig.add_subplot(224)
    data0 = c0.statistics['vel_max(t)'] * (c0.parameters['dt'] * c0.parameters['dkx'] /
                                           (2*np.pi / c0.parameters['nx']))
    data1 = c1.statistics['vel_max(t)'] * (c1.parameters['dt'] * c1.parameters['dkx'] /
                                           (2*np.pi / c1.parameters['nx']))
    a.plot(c0.statistics['t'],
           data0 - data1)
    a.set_title('$\\frac{\\Delta t \\| u \\|_\infty}{\\Delta x}$')
    fig.savefig('plain_stat_diffs_{0}.pdf'.format(opt.precision), format = 'pdf')

    # plot trajectory differences
    for i in range(c0.particle_species):
        fig = plt.figure(figsize=(12, 4))
        for j in range(3):
            a = fig.add_subplot(131 + j)
            for t in range(c0.parameters['nparticles']):
                a.plot(c0.trajectories[i][:, t, j] - c1.trajectories[i][:, t, j])
        fig.savefig('traj_s{0}_{1}.pdf'.format(i, opt.precision), format = 'pdf')
    return None

if __name__ == '__main__':
    opt = parser.parse_args()
    plain(opt)

