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

def plain(opt):
    wd = opt.work_dir
    opt.work_dir = wd + '/N{0:0>3x}_1'.format(opt.n)
    c0 = NSlaunch(opt)
    c0.compute_statistics()
    opt.work_dir = wd + '/N{0:0>3x}_2'.format(opt.n)
    opt.njobs *= 2
    opt.nsteps /= 2
    c1 = NSlaunch(opt)
    c1.compute_statistics()
    # plot energy and enstrophy
    fig = plt.figure(figsize = (12, 12))
    a = fig.add_subplot(221)
    c0.set_plt_style({'label' : '1',
                      'dashes' : (None, None),
                      'color' : (1, 0, 0)})
    c1.set_plt_style({'label' : '2',
                      'dashes' : (2, 2),
                      'color' : (0, 0, 1)})
    for c in [c0, c1]:
        a.plot(c.statistics['t'],
               c.statistics['energy(t)'],
               label = c.style['label'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('energy')
    a.legend(loc = 'best')
    a = fig.add_subplot(222)
    for c in [c0, c1]:
        a.plot(c.statistics['t'],
               c.statistics['enstrophy(t)'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('enstrophy')
    a = fig.add_subplot(223)
    for c in [c0, c1]:
        a.plot(c.statistics['t'],
               c.statistics['kM']*c.statistics['etaK(t)'],
               dashes = c.style['dashes'],
               color = c.style['color'])
    a.set_title('$k_M \\eta_K$')
    a = fig.add_subplot(224)
    for c in [c0, c1]:
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
