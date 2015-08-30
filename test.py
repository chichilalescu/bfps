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


import os
import subprocess
import argparse
import pickle

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import h5py

import bfps
from bfps import fluid_resize

def Kolmogorov_flow_test_broken(opt):
    c = convergence_test(name = 'Kflow_test')
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 1.0
    c.parameters['dt'] = 0.02
    c.parameters['niter_todo'] = opt.nsteps
    if opt.run:
        np.random.seed(7547)
        Kdata00 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata01 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata02 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        c.fill_up_fluid_code()
        c.finalize_code()
        c.write_src()
        c.write_par()
        Kdata0.tofile("test_cvorticity_i00000")
        c.run(ncpu = opt.ncpu)
        Rdata = np.fromfile(
                'test_rvorticity_i00000',
                dtype = np.float32).reshape(opt.n, opt.n, opt.n, 3)
        tdata = Rdata.transpose(3, 0, 1, 2).copy()
    dtype = pickle.load(open(c.name + '_dtype.pickle'))
    stats = np.fromfile('test_stats.bin', dtype = dtype)
    fig = plt.figure(figsize = (12,6))
    a = fig.add_subplot(121)
    a.plot(stats['t'], stats['energy'])
    a = fig.add_subplot(122)
    a.plot(stats['t'], stats['enstrophy'])
    fig.savefig('test.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(221)
    c.plot_vel_cut(a,
            field = 'velocity',
            iteration = 0)
    a = fig.add_subplot(222)
    a.set_axis_off()
    c.plot_vel_cut(a,
            field = 'vorticity',
            iteration = 0)
    a = fig.add_subplot(223)
    ufin = c.plot_vel_cut(a,
            field = 'velocity',
            iteration = stats.shape[0]-1)
    a = fig.add_subplot(224)
    vfin = c.plot_vel_cut(a,
            field = 'vorticity',
            iteration = stats.shape[0]-1)
    fig.savefig('field_cut.pdf', format = 'pdf')

    fig = plt.figure()
    a = fig.add_subplot(111)
    ycoord = c.get_coord('y')
    a.plot(ycoord, ufin[0, :, 0, 0])
    amp = c.parameters['famplitude'] / (c.parameters['fmode']**2 * c.parameters['nu'])
    a.plot(ycoord, amp*np.sin(ycoord), dashes = (2, 2))
    a.plot(ycoord, vfin[0, :, 0, 2])
    a.plot(ycoord, -np.cos(ycoord), dashes = (2, 2))
    a.set_xticks(np.linspace(.0, 2*np.pi, 9))
    a.set_xlim(.0, 2*np.pi)
    a.grid()
    fig.savefig('ux_vs_y.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 12))
    tmp = np.fromfile('test_cvelocity_i{0:0>5x}'.format(stats.shape[0]-1),
                      dtype = np.complex64).reshape(opt.n, opt.n, opt.n/2+1, 3)
    a = fig.add_subplot(321)
    a.plot(np.sum(np.abs(tmp), axis = (1, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(323)
    a.plot(np.sum(np.abs(tmp), axis = (0, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(325)
    a.plot(np.sum(np.abs(tmp), axis = (0, 1)))
    a.set_yscale('log')
    a = fig.add_subplot(322)
    tmp = np.fromfile('test_cvorticity_i{0:0>5x}'.format(stats.shape[0]-1),
                      dtype = np.complex64).reshape(opt.n, opt.n, opt.n/2+1, 3)
    a.plot(np.sum(np.abs(tmp), axis = (1, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(324)
    a.plot(np.sum(np.abs(tmp), axis = (0, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(326)
    a.plot(np.sum(np.abs(tmp), axis = (0, 1)))
    a.set_yscale('log')
    fig.savefig('cvort.pdf', format = 'pdf')
    return None

def double(opt):
    old_simname = 'N{0:0>3x}'.format(opt.n)
    new_simname = 'N{0:0>3x}'.format(opt.n*2)
    c = fluid_resize(
            work_dir = opt.work_dir + '/resize',
            simname = old_simname,
            dtype = opt.precision)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = new_simname
    c.write_src()
    c.set_host_info({'type' : 'pc'})
    if not os.path.isdir(os.path.join(opt.work_dir, 'resize')):
        os.mkdir(os.path.join(opt.work_dir, 'resize'))
    c.write_par()
    cp_command = ('cp {0}/test_cvorticity_i{1:0>5x} {2}/{3}_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + old_simname, opt.iteration, opt.work_dir + '/resize', old_simname))
    subprocess.call([cp_command], shell = True)
    c.run(ncpu = opt.ncpu,
          err_file = 'err_' + old_simname + '_' + new_simname,
          out_file = 'out_' + old_simname + '_' + new_simname)
    if not os.path.isdir(os.path.join(opt.work_dir, new_simname)):
        os.mkdir(os.path.join(opt.work_dir, new_simname))
    cp_command = ('cp {2}/{3}_cvorticity_i{1:0>5x} {0}/test_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + new_simname, 0, opt.work_dir + '/resize', new_simname))
    subprocess.call([cp_command], shell = True)
    return None

def NSlaunch(
        opt,
        nu = None,
        tracer_state_file = None):
    c = bfps.NavierStokes(
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
    c.parameters['dt'] = 5e-3 * (64. / opt.n)
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['niter_out'] = opt.nsteps
    c.parameters['niter_part'] = 1
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = 8
    c.add_particles(kcut = 'fs->kM/2')
    c.add_particles(integration_steps = 1)
    c.add_particles(integration_steps = 2)
    c.add_particles(integration_steps = 3)
    c.add_particles(integration_steps = 4)
    c.add_particles(integration_steps = 5)
    c.add_particles(integration_steps = 6)
    c.fill_up_fluid_code()
    c.finalize_code()
    c.write_src()
    c.write_par()
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            c.generate_vector_field(write_to_file = True)
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

def convergence_test(opt):
    ### test Navier Stokes convergence
    # first, run code three times, doubling and quadrupling the resolution
    # initial condition and viscosity must be the same!
    default_wd = opt.work_dir
    opt.work_dir = default_wd + '/N{0:0>3x}'.format(opt.n)
    c0 = NSlaunch(opt)
    opt.initialize = False
    opt.work_dir = default_wd
    double(opt)
    opt.iteration = 0
    opt.n *= 2
    opt.nsteps *= 2
    opt.ncpu *= 2
    opt.work_dir = default_wd + '/N{0:0>3x}'.format(opt.n)
    c1 = NSlaunch(
            opt,
            nu = c0.parameters['nu'],
            tracer_state_file = h5py.File(os.path.join(c0.work_dir, c0.simname + '.h5'), 'r'))
    opt.work_dir = default_wd
    double(opt)
    opt.n *= 2
    opt.nsteps *= 2
    opt.ncpu *= 2
    opt.work_dir = default_wd + '/N{0:0>3x}'.format(opt.n)
    c2 = NSlaunch(
            opt,
            nu = c0.parameters['nu'],
            tracer_state_file = h5py.File(os.path.join(c0.work_dir, c0.simname + '.h5'), 'r'))
    # get real space fields
    converter = bfps.fluid_converter()
    converter.write_src()
    converter.set_host_info({'type' : 'pc'})
    for c in [c0, c1, c2]:
        converter.work_dir = c.work_dir
        converter.simname = c.simname + '_converter'
        for key in converter.parameters.keys():
            if key in c.parameters.keys():
                converter.parameters[key] = c.parameters[key]
        converter.parameters['fluid_name'] = c.simname
        converter.write_par()
        converter.run(
                ncpu = 2)
        c.transpose_frame(iteration = c.parameters['niter_todo'])
    # read data
    c0.compute_statistics()
    c0.set_plt_style({'dashes': (None, None)})
    c1.compute_statistics()
    c1.set_plt_style({'dashes': (2, 3)})
    c2.compute_statistics()
    c2.set_plt_style({'dashes': (3, 4)})
    for c in [c0, c1, c2]:
        c.style.update({'label' : '${0}\\times {1} \\times {2}$'.format(c.parameters['nx'],
                                                                        c.parameters['ny'],
                                                                        c.parameters['nz'])})
    # plot slices
    def plot_face_contours(axis, field, levels = None):
        xx, yy = np.meshgrid(np.linspace(0, 1, field.shape[1]),
                             np.linspace(0, 1, field.shape[2]))
        if type(levels) == type(None):
            emin = np.min(field)
            emax = np.max(field)
            levels = np.linspace(emin + (emax - emin)/20,
                                 emax - (emax - emin)/20,
                                 20)
        cz = axis.contour(xx, yy, field[0],       zdir = 'z', offset = 0.0, levels = levels)
        xx, yy = np.meshgrid(np.linspace(0, 1, field.shape[0]),
                             np.linspace(0, 1, field.shape[2]))
        cy = axis.contour(xx, field[:, 0], yy,    zdir = 'y', offset = 1.0, levels = levels)
        xx, yy = np.meshgrid(np.linspace(0, 1, field.shape[0]),
                             np.linspace(0, 1, field.shape[1]))
        cx = axis.contour(field[:, :, 0], xx, yy, zdir = 'x', offset = 0.0, levels = levels)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_zlim(0, 1)
        return levels
    def full_face_contours_fig(field_name = 'velocity'):
        fig = plt.figure(figsize = (18,6))
        a = fig.add_subplot(131, projection = '3d')
        vec = c0.read_rfield(iteration = c0.parameters['niter_todo'], field = field_name)
        levels = plot_face_contours(a, .5*np.sum(vec**2, axis = 3))
        a.set_title(c0.style['label'])
        a = fig.add_subplot(132, projection = '3d')
        vec = c1.read_rfield(iteration = c1.parameters['niter_todo'], field = field_name)
        plot_face_contours(a, .5*np.sum(vec**2, axis = 3), levels = levels)
        a.set_title(c1.style['label'])
        a = fig.add_subplot(133, projection = '3d')
        vec = c2.read_rfield(iteration = c2.parameters['niter_todo'], field = field_name)
        plot_face_contours(a, .5*np.sum(vec**2, axis = 3), levels = levels)
        a.set_title(c2.style['label'])
        fig.savefig(field_name + '_contour_' + opt.precision + '.pdf', format = 'pdf')
    full_face_contours_fig()
    full_face_contours_fig(field_name = 'vorticity')
    # plot spectra
    def plot_spec(a, c):
        for i in range(c.statistics['energy(t, k)'].shape[0]):
            a.plot(c.statistics['kshell'],
                   c.statistics['energy(t, k)'][i],
                   color = plt.get_cmap('coolwarm')(i*1.0/(c.statistics['energy(t, k)'].shape[0])))
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_title(c.style['label'])
    fig = plt.figure(figsize=(12, 4))
    plot_spec(fig.add_subplot(131), c0)
    plot_spec(fig.add_subplot(132), c1)
    plot_spec(fig.add_subplot(133), c2)
    fig.savefig('spectra_' + opt.precision + '.pdf', format = 'pdf')
    # plot energy and enstrophy
    fig = plt.figure(figsize = (12, 12))
    a = fig.add_subplot(221)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['energy(t)'],
               label = c.style['label'],
               dashes = c.style['dashes'])
    a.set_title('energy')
    a.legend(loc = 'best')
    a = fig.add_subplot(222)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['enstrophy(t)'],
               dashes = c.style['dashes'])
    a.set_title('enstrophy')
    a = fig.add_subplot(223)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['kM']*c.statistics['etaK(t)'],
               dashes = c.style['dashes'])
    a.set_title('$k_M \\eta_K$')
    a = fig.add_subplot(224)
    for c in [c0, c1, c2]:
        a.plot(c.statistics['t'],
               c.statistics['vel_max(t)'] * (c.parameters['dt'] * c.parameters['dkx'] /
                                             (2*np.pi / c.parameters['nx'])),
               dashes = c.style['dashes'])
    a.set_title('$\\frac{\\Delta t \\| u \\|_\infty}{\\Delta x}$')
    fig.savefig('convergence_stats_' + opt.precision + '.pdf', format = 'pdf')
    ## particle test:
    # compute distance between final positions for species 1
    def get_traj_error(species):
        e0 = np.abs(c0.trajectories[species][-1, :, :3] - c1.trajectories[species][-1, :, :3])
        e1 = np.abs(c1.trajectories[species][-1, :, :3] - c2.trajectories[species][-1, :, :3])
        return np.array([np.average(np.sqrt(np.sum(e0**2, axis = 1))),
                         np.average(np.sqrt(np.sum(e1**2, axis = 1)))])
    err = [get_traj_error(i) for i in range(1, c0.particle_species)]
    fig = plt.figure()
    a = fig.add_subplot(111)
    for i in range(1, c0.particle_species):
        print('{0} {1}'.format(i, err[i-1]))
        a.plot([c0.parameters['dt'], c1.parameters['dt']],
               err[i-1],
               marker = '.',
               label = '${0}$'.format(i))
    a.plot( [c0.parameters['dt'], c1.parameters['dt']],
            [c0.parameters['dt'], c1.parameters['dt']],
            label = '$\\Delta t$',
            dashes = (1,1),
            color = (0, 0, 0))
    a.set_xscale('log')
    a.set_yscale('log')
    a.legend(loc = 'best')
    fig.savefig('traj_evdt_' + opt.precision + '.pdf', format = 'pdf')
    # plot all trajectories... just in case
    for c in [c0, c1, c2]:
        fig = plt.figure(figsize=(12,12))
        a = fig.add_subplot(111, projection = '3d')
        for t in range(c.parameters['nparticles']):
            for i in range(1, c.particle_species):
                a.plot(c.trajectories[i][:, t, 0],
                       c.trajectories[i][:, t, 1],
                       c.trajectories[i][:, t, 2])
        fig.savefig('traj_N{0:0>3x}_{1}.pdf'.format(c.parameters['nx'], opt.precision), format = 'pdf')
    return None

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
    a.legend(loc = 'best')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('--initialize', dest = 'initialize', action = 'store_true')
    parser.add_argument('--convergence', dest = 'convergence', action = 'store_true')
    parser.add_argument('--plain', dest = 'plain', action = 'store_true')
    parser.add_argument('--io', dest = 'io', action = 'store_true')
    parser.add_argument('-n',
            type = int, dest = 'n', default = 64)
    parser.add_argument('--iteration',
            type = int, dest = 'iteration', default = 0)
    parser.add_argument('--ncpu',
            type = int, dest = 'ncpu', default = 2)
    parser.add_argument('--nsteps',
            type = int, dest = 'nsteps', default = 16)
    parser.add_argument('--njobs',
            type = int, dest = 'njobs', default = 1)
    parser.add_argument('--wd',
            type = str, dest = 'work_dir', default = 'data')
    parser.add_argument('--precision',
            type = str, dest = 'precision', default = 'single')
    opt = parser.parse_args()
    if opt.convergence:
        convergence_test(opt)
    if opt.plain:
        plain(opt)
    if opt.io:
        c = bfps.test_io(work_dir = opt.work_dir + '/test_io')
        c.write_src()
        c.write_par()
        c.set_host_info({'type' : 'pc'})
        c.run(ncpu = opt.ncpu)

