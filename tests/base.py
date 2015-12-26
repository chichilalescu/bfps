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
from bfps import fluid_resize

parser = bfps.get_parser()
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
    c = fluid_resize(
            work_dir = opt.work_dir,
            simname = old_simname + '_double',
            dtype = opt.precision)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = new_simname
    c.parameters['src_simname'] = old_simname
    c.parameters['niter_todo'] = 0
    c.write_src()
    c.set_host_info({'type' : 'pc'})
    c.write_par()
    c.run(ncpu = opt.ncpu,
          err_file = 'err_',
          out_file = 'out_')
    return None

def launch(
        opt,
        nu = None,
        dt = None,
        tracer_state_file = None,
        vorticity_field = None,
        code_class = bfps.NavierStokes):
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
    c.parameters['niter_part'] = 1
    c.parameters['famplitude'] = 0.2
    c.fill_up_fluid_code()
    if c.parameters['nparticles'] > 0:
        c.add_particle_fields(
                name = 'regular',
                neighbours = opt.neighbours,
                smoothness = opt.smoothness)
        c.add_particle_fields(kcut = 'fs->kM/2', name = 'filtered', neighbours = opt.neighbours)
        c.add_particles(
                kcut = 'fs->kM/2',
                integration_steps = 1,
                fields_name = 'filtered')
        #for integr_steps in range(1, 7):
        #    c.add_particles(
        #            integration_steps = integr_steps,
        #            neighbours = opt.neighbours,
        #            smoothness = opt.smoothness,
        #            fields_name = 'regular')
        for info in [(2, 'AdamsBashforth'),
                     (3, 'AdamsBashforth'),
                     (4, 'AdamsBashforth'),
                     (6, 'AdamsBashforth')]:
            c.add_particles(
                    integration_steps = info[0],
                    integration_method = info[1],
                    fields_name = 'regular')
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


def acceleration_test(c, m = 3, species = 0):
    import numpy as np
    import matplotlib.pyplot as plt
    from bfps.tools import get_fornberg_coeffs
    d = c.get_data_file()
    group = d['particles/tracers{0}'.format(species)]
    pos = group['state'].value
    vel = group['velocity'].value
    acc = group['acceleration'].value
    fig = plt.figure()
    a = fig.add_subplot(111)
    col = ['red', 'green', 'blue']
    n = m
    fc = get_fornberg_coeffs(0, range(-n, n+1))
    dt = d['parameters/dt'].value*d['parameters/niter_part'].value

    num_acc1 = sum(fc[1, n-i]*vel[1+n-i:vel.shape[0]-i-n-1] for i in range(-n, n+1)) / dt
    num_acc2 = sum(fc[2, n-i]*pos[1+n-i:pos.shape[0]-i-n-1] for i in range(-n, n+1)) / dt**2
    num_vel1 = sum(fc[1, n-i]*pos[1+n-i:pos.shape[0]-i-n-1] for i in range(-n, n+1)) / dt

    def SNR(a, b):
        return -10*np.log10(np.mean((a - b)**2, axis = (0, 2)) / np.mean(a**2, axis = (0, 2)))
    pid = np.argmin(SNR(num_acc1, acc[n+1:-n-1]))
    pars = d['parameters']
    to_print = (
            'integration={0}, steps={1}, interp={2}, neighbours={3}, '.format(
                pars['tracers{0}_integration_method'.format(species)].value,
                pars['tracers{0}_integration_steps'.format(species)].value,
                pars[str(pars['tracers{0}_field'.format(species)].value) + '_type'].value,
                pars[str(pars['tracers{0}_field'.format(species)].value) + '_neighbours'].value))
    if 'spline' in pars['tracers{0}_field'.format(species)].value:
        to_print += 'smoothness = {0}, '.format(pars[str(pars['tracers{0}_field'.format(species)].value) + '_smoothness'].value)
    to_print += (
            'SNR d1p-vel={0:.3f}, d1v-acc={1:.3f}, d2p-acc={2:.3f}'.format(
                np.mean(SNR(num_vel1, vel[n+1:-n-1])),
                np.mean(SNR(num_acc1, acc[n+1:-n-1])),
                np.mean(SNR(num_acc2, acc[n+1:-n-1]))))
    print(to_print)
    for cc in range(3):
        a.plot(num_acc1[:, pid, cc], color = col[cc])
        a.plot(num_acc2[:, pid, cc], color = col[cc], dashes = (2, 2))
        a.plot(acc[m+1:, pid, cc], color = col[cc], dashes = (1, 1))

    for n in range(1, m):
        fc = get_fornberg_coeffs(0, range(-n, n+1))
        dt = d['parameters/dt'].value*d['parameters/niter_part'].value

        num_acc1 = sum(fc[1, n-i]*vel[n-i:vel.shape[0]-i-n] for i in range(-n, n+1)) / dt
        num_acc2 = sum(fc[2, n-i]*pos[n-i:pos.shape[0]-i-n] for i in range(-n, n+1)) / dt**2

        for cc in range(3):
            a.plot(num_acc1[m-n:, pid, cc], color = col[cc])
            a.plot(num_acc2[m-n:, pid, cc], color = col[cc], dashes = (2, 2))
    fig.tight_layout()
    fig.savefig('acc_test_{0}_{1}.pdf'.format(c.simname, species))
    plt.close(fig)
    return pid

def compare_stats(
        opt,
        c0, c1,
        plots_on = False):
    for key in ['energy', 'enstrophy', 'vel_max']:
        print('maximum {0} difference is {1}'.format(
            key,
            np.max(np.abs(c0.statistics[key + '(t)'] - c0.statistics[key + '(t)']))))
    for i in range(c0.particle_species):
        print('maximum traj difference species {0} is {1}'.format(
            i,
            np.max(np.abs(c0.trajectories[i] - c1.trajectories[i]))))
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

