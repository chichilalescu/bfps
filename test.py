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


import numpy as np
import subprocess
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse
import pickle
import os

import bfps
from bfps.NavierStokes import launch as NSlaunch
from bfps.resize import vorticity_resize
from bfps.test_curl import test as test_curl

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
        c.write_src()
        c.write_par(simname = 'test')
        Kdata0.tofile("test_cvorticity_i00000")
        c.run(ncpu = opt.ncpu, simname = 'test')
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
            simname = 'test',
            field = 'velocity',
            iteration = 0)
    a = fig.add_subplot(222)
    a.set_axis_off()
    c.plot_vel_cut(a,
            simname = 'test',
            field = 'vorticity',
            iteration = 0)
    a = fig.add_subplot(223)
    ufin = c.plot_vel_cut(a,
            simname = 'test',
            field = 'velocity',
            iteration = stats.shape[0]-1)
    a = fig.add_subplot(224)
    vfin = c.plot_vel_cut(a,
            simname = 'test',
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
    c = vorticity_resize(work_dir = opt.work_dir + '/resize')
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = new_simname
    c.write_src()
    if not os.path.isdir(os.path.join(opt.work_dir, 'resize')):
        os.mkdir(os.path.join(opt.work_dir, 'resize'))
    c.write_par(simname = old_simname)
    cp_command = ('cp {0}/test_cvorticity_i{1:0>5x} {2}/{3}_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + old_simname, opt.iteration, opt.work_dir + '/resize', old_simname))
    subprocess.call([cp_command], shell = True)
    c.run(ncpu = opt.ncpu,
          simname = old_simname,
          iter0 = opt.iteration)
    if not os.path.isdir(os.path.join(opt.work_dir, new_simname)):
        os.mkdir(os.path.join(opt.work_dir, new_simname))
    cp_command = ('cp {2}/{3}_cvorticity_i{1:0>5x} {0}/test_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + new_simname, 0, opt.work_dir + '/resize', new_simname))
    subprocess.call([cp_command], shell = True)
    np.array([0.0]).tofile(
            os.path.join(
                    opt.work_dir + '/' + new_simname, 'test_time_i00000'))
    return None

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
    cp_command = ('cp {0}/test_tracers?_state_i00000 {1}/'.format(c0.work_dir, opt.work_dir))
    subprocess.call([cp_command], shell = True)
    c1 = NSlaunch(
            opt,
            nu = c0.parameters['nu'])
    opt.work_dir = default_wd
    double(opt)
    opt.n *= 2
    opt.nsteps *= 2
    opt.ncpu *= 2
    opt.work_dir = default_wd + '/N{0:0>3x}'.format(opt.n)
    cp_command = ('cp {0}/test_tracers?_state_i00000 {1}/'.format(c0.work_dir, opt.work_dir))
    subprocess.call([cp_command], shell = True)
    c2 = NSlaunch(
            opt,
            nu = c0.parameters['nu'])
    ## fluid test:
    # read data
    c0.compute_statistics()
    c0.set_plt_style(
            {'dashes': (None, None),
             'label' : '${0}\\times {1} \\times {2}$'.format(c0.parameters['nx'],
                                                             c0.parameters['ny'],
                                                             c0.parameters['nz'])})
    c1.compute_statistics()
    c1.set_plt_style(
            {'dashes': (2, 3),
             'label' : '${0}\\times {1} \\times {2}$'.format(c1.parameters['nx'],
                                                             c1.parameters['ny'],
                                                             c1.parameters['nz'])})
    c2.compute_statistics()
    c2.set_plt_style(
            {'dashes': (3, 4),
             'label' : '${0}\\times {1} \\times {2}$'.format(c2.parameters['nx'],
                                                             c2.parameters['ny'],
                                                             c2.parameters['nz'])})
    # plot spectra
    def plot_spec(a, c):
        for i in range(c.statistics['energy(t, k)'].shape[0]):
            a.plot(c.statistics['k'],
                   c.statistics['energy(t, k)'][i],
                   color = plt.get_cmap('coolwarm')(i*1.0/(c.statistics['energy(t, k)'].shape[0])))
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_title(c.style['label'])
    fig = plt.figure(figsize=(12, 4))
    plot_spec(fig.add_subplot(131), c0)
    plot_spec(fig.add_subplot(132), c1)
    plot_spec(fig.add_subplot(133), c2)
    fig.savefig('spectra.pdf', format = 'pdf')
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
    fig.savefig('stats.pdf', format = 'pdf')
    ## particle test:
    # compute distance between final positions for species 0
    e0 = np.abs(c0.trajectories['state'][0, -1, :, :3] - c1.trajectories['state'][0, -1, :, :3])
    e1 = np.abs(c1.trajectories['state'][0, -1, :, :3] - c2.trajectories['state'][0, -1, :, :3])
    err0 = np.average(np.sqrt(np.sum(e0**2, axis = 1)))
    err1 = np.average(np.sqrt(np.sum(e0**2, axis = 1)))
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.plot([c0.parameters['dt'], c1.parameters['dt']],
           [err0, err1],
           marker = '.')
    a.set_xscale('log')
    a.set_yscale('log')
    fig.savefig('traj_evdt.pdf', format = 'pdf')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--particles', dest = 'particles', action = 'store_true')
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('--double', dest = 'double', action = 'store_true')
    parser.add_argument('--initialize', dest = 'initialize', action = 'store_true')
    parser.add_argument('--convergence', dest = 'convergence', action = 'store_true')
    parser.add_argument('--iteration', type = int, dest = 'iteration', default = 0)
    parser.add_argument('--ncpu', type = int, dest = 'ncpu', default = 2)
    parser.add_argument('--nsteps', type = int, dest = 'nsteps', default = 16)
    parser.add_argument('--njobs', type = int, dest = 'njobs', default = 1)
    parser.add_argument('-n', type = int, dest = 'n', default = 64)
    parser.add_argument('--wd', type = str, dest = 'work_dir', default = 'data')
    opt = parser.parse_args()
    if opt.convergence:
        convergence_test(opt)
    else:
        opt.work_dir += '/N{0:0>3x}'.format(opt.n)
        c0 = NSlaunch(opt)
        c0.compute_statistics()
        c0.basic_plots()

