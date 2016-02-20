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



import os
import sys
import subprocess
import pickle

import numpy as np
import matplotlib.pyplot as plt

import bfps
from bfps import FluidResize
from bfps.tools import particle_finite_diff_test as acceleration_test

import argparse

def get_parser(base_class = bfps.NavierStokes,
               n = 32,
               ncpu = 2,
               precision = 'single',
               simname = 'test',
               work_dir = './',
               njobs = 1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('-n',
            type = int, dest = 'n',
            default = n)
    parser.add_argument('--ncpu',
            type = int, dest = 'ncpu',
            default = ncpu)
    parser.add_argument('--precision',
            type = str, dest = 'precision',
            default = precision)
    parser.add_argument('--simname',
            type = str, dest = 'simname',
            default = simname)
    parser.add_argument('--wd',
            type = str, dest = 'work_dir',
            default = work_dir)
    parser.add_argument('--njobs',
            type = int, dest = 'njobs',
            default = njobs)
    c = base_class(simname = simname)
    for k in sorted(c.parameters.keys()):
        parser.add_argument(
                '--{0}'.format(k),
                type = type(c.parameters[k]),
                dest = k,
                default = c.parameters[k])
    return parser

parser = get_parser()
parser.add_argument('--initialize', dest = 'initialize', action = 'store_true')
parser.add_argument('--frozen', dest = 'frozen', action = 'store_true')
parser.add_argument('--iteration',
        type = int, dest = 'iteration', default = 0)
parser.add_argument('--neighbours',
        type = int, dest = 'neighbours', default = 3)
parser.add_argument('--smoothness',
        type = int, dest = 'smoothness', default = 2)
parser.add_argument(
        '--kMeta',
        type = float,
        dest = 'kMeta',
        default = 2.0)

def double(opt):
    old_simname = 'N{0:0>3x}'.format(opt.n)
    new_simname = 'N{0:0>3x}'.format(opt.n*2)
    c = FluidResize(fluid_precision = opt.precision)
    c.launch(
            args = ['--simname', old_simname + '_double',
                    '--wd', opt.work_dir,
                    '--nx', '{0}'.format(opt.n),
                    '--ny', '{0}'.format(opt.n),
                    '--nz', '{0}'.format(opt.n),
                    '--dst_nx', '{0}'.format(2*opt.n),
                    '--dst_ny', '{0}'.format(2*opt.n),
                    '--dst_nz', '{0}'.format(2*opt.n),
                    '--dst_simname', new_simname,
                    '--src_simname', old_simname,
                    '--src_iteration', '0',
                    '--src_wd', './',
                    '--niter_todo', '0'])
    return None

def launch(
        opt,
        nu = None,
        dt = None,
        tracer_state_file = None,
        vorticity_field = None,
        code_class = bfps.NavierStokes,
        particle_class = 'particles',
        interpolator_class = 'rFFTW_interpolator'):
    c = code_class(
            work_dir = opt.work_dir,
            fluid_precision = opt.precision,
            frozen_fields = opt.frozen,
            use_fftw_wisdom = False)
    if code_class == bfps.NavierStokes:
        c.QR_stats_on = True
    c.pars_from_namespace(opt)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
    if type(dt) == type(None):
        c.parameters['dt'] = (0.5 / opt.n)
    else:
        c.parameters['dt'] = dt
    c.parameters['niter_out'] = c.parameters['niter_todo']
    c.parameters['famplitude'] = 0.2
    c.fill_up_fluid_code()
    if c.parameters['nparticles'] > 0:
        c.name += '-' + particle_class
        c.add_3D_rFFTW_field(name = 'rFFTW_acc')
        c.add_interpolator(
                name = 'spline',
                neighbours = opt.neighbours,
                smoothness = opt.smoothness,
                class_name = interpolator_class)
        c.add_particles(
                kcut = ['fs->kM/2', 'fs->kM/3'],
                integration_steps = 3,
                interpolator = 'spline',
                class_name = particle_class)
        c.add_particles(
                integration_steps = [2, 3, 4, 6],
                interpolator = 'spline',
                acc_name = 'rFFTW_acc',
                class_name = particle_class)
    c.finalize_code()
    c.write_src()
    c.write_par()
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            if type(vorticity_field) == type(None):
                c.generate_vector_field(write_to_file = True,
                                        spectra_slope = 2.0,
                                        amplitude = 0.25)
            else:
                vorticity_field.tofile(
                        os.path.join(c.work_dir,
                                     c.simname + "_c{0}_i{1:0>5x}".format('vorticity',
                                                                          opt.iteration)))
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

def compare_stats(
        opt,
        c0, c1,
        plots_on = False):
    for key in ['energy', 'enstrophy', 'vel_max']:
        print('maximum {0} difference is {1}'.format(
            key,
            np.max(np.abs(c0.statistics[key + '(t)'] - c0.statistics[key + '(t)']))))
    for i in range(c0.particle_species):
        print('species={0} differences: max pos(t) {1:.3e} min vel(0) {2:.3e}, max vel(0) {3:.3e}'.format(
            i,
            np.max(np.abs(c0.get_particle_file()['tracers{0}/state'.format(i)][:] -
                          c1.get_particle_file()['tracers{0}/state'.format(i)][:])),
            np.min(np.abs(c0.get_particle_file()['tracers{0}/velocity'.format(i)][0] -
                          c1.get_particle_file()['tracers{0}/velocity'.format(i)][0])),
            np.max(np.abs(c0.get_particle_file()['tracers{0}/velocity'.format(i)][0] -
                          c1.get_particle_file()['tracers{0}/velocity'.format(i)][0]))))
    if plots_on:
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
    print('this file doesn\'t do anything')

