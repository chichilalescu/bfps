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


from bfps.test import convergence_test
from bfps.NavierStokes import test as NStest
from bfps.resize import double as resize_test
from bfps.test_curl import test as test_curl

import numpy as np
import subprocess
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse
import pickle

def main(opt):
    if opt.run or opt.clean:
        subprocess.call(['rm test1_*'], shell = True)
        subprocess.call(['rm test2_*'], shell = True)
        subprocess.call(['rm input_split_per_component'], shell = True)
        subprocess.call(['rm *.pickle'], shell = True)
    if opt.clean:
        subprocess.call(['make', 'clean'])
        return None
    c = convergence_test()
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 1e-1
    c.parameters['dt'] = 2e-3
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['famplitude'] = 0.0
    c.parameters['nparticles'] = 32
    if opt.run:
        c.execute(
                ncpu = opt.ncpu)#,
#                tracer_state = np.array([1.56292, 2.33185, 1.42517]))
    dtype = pickle.load(open(c.name + '_dtype.pickle'))
    stats1 = np.fromfile('test1_stats.bin', dtype = dtype)
    stats2 = np.fromfile('test2_stats.bin', dtype = dtype)
    stats_vortex = np.loadtxt('../vortex/sim_000000.log')
    traj1 = np.fromfile('test1_tracers_traj.bin', dtype = np.float64).reshape(-1, c.parameters['nparticles'], 3)
    traj2 = np.fromfile('test2_tracers_traj.bin', dtype = np.float64).reshape(-1, c.parameters['nparticles'], 3)
    fig = plt.figure(figsize = (12,6))
    a = fig.add_subplot(121)
    a.plot(stats1['t'], stats1['energy'])
    a.plot(stats2['t'], stats2['energy'], dashes = (3, 3))
    a.plot(stats_vortex[:, 2], stats_vortex[:, 3], dashes = (2, 4))
    a = fig.add_subplot(122)
    a.plot(stats1['t'], stats1['enstrophy'])
    a.plot(stats2['t'], stats2['enstrophy'], dashes = (3, 3))
    a.plot(stats_vortex[:, 2], stats_vortex[:, 9]/2, dashes = (2, 4))
    fig.savefig('test.pdf', format = 'pdf')

    fig = plt.figure(figsize = (24, 12))
    a = fig.add_subplot(121, projection = '3d')
    for t in range(traj1.shape[1]):
        a.plot(traj1[:, t, 0], traj1[:, t, 1], traj1[:, t, 2], color = 'blue')
        a.plot(traj2[:, t, 0], traj2[:, t, 1], traj2[:, t, 2], color = 'red', dashes = (1,1))
    a = fig.add_subplot(322)
    for t in range(traj1.shape[1]):
        a.plot(stats1['t'], traj1[:, t, 0], color = 'blue')
        a.plot(stats2['t'], traj2[:, t, 0], color = 'red', dashes = (1,1))
    a = fig.add_subplot(324)
    for t in range(traj1.shape[1]):
        a.plot(stats1['t'], traj1[:, t, 1], color = 'blue')
        a.plot(stats2['t'], traj2[:, t, 1], color = 'red', dashes = (1,1))
    a = fig.add_subplot(326)
    for t in range(traj1.shape[1]):
        a.plot(stats1['t'], traj1[:, t, 2], color = 'blue')
        a.plot(stats2['t'], traj2[:, t, 2], color = 'red', dashes = (1,1))
    fig.savefig('traj.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(221)
    c.plot_vel_cut(
            a,
            simname = 'test1',
            field = 'rvelocity',
            iteration = 0)
    a = fig.add_subplot(223)
    c.plot_vel_cut(
            a,
            simname = 'test1',
            field = 'rvelocity',
            iteration = stats1.shape[0] - 1)
    a = fig.add_subplot(222)
    c.parameters['nx'] *= 2
    c.parameters['ny'] *= 2
    c.parameters['nz'] *= 2
    c.plot_vel_cut(
            a,
            simname = 'test2',
            field = 'rvelocity',
            iteration = 0,
            yval = 26)
    a = fig.add_subplot(224)
    c.plot_vel_cut(
            a,
            simname = 'test2',
            field = 'rvelocity',
            iteration = stats2.shape[0] - 1,
            yval = 26)
    fig.savefig('vel_cut.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 6))
    a = fig.add_subplot(121)
    c.parameters['nx'] /= 2
    c.parameters['ny'] /= 2
    c.parameters['nz'] /= 2
    c.plot_vel_cut(
            a,
            simname = 'test1',
            field = 'particle_field',
            iteration = 0,
            yval = 13)
    a = fig.add_subplot(122)
    c.parameters['nx'] *= 2
    c.parameters['ny'] *= 2
    c.parameters['nz'] *= 2
    c.plot_vel_cut(
            a,
            simname = 'test2',
            field = 'particle_field',
            iteration = 0,
            yval = 26)
    fig.savefig('particle_field.pdf', format = 'pdf')

    k1 = np.fromfile('test1_kshell', dtype = np.float64)
    k2 = np.fromfile('test2_kshell', dtype = np.float64)
    fig = plt.figure(figsize=(12, 6))
    a = fig.add_subplot(121)
    s1 = np.fromfile('test1_velocity_spec_i{0:0>5x}'.format(0), dtype = np.float64)
    s2 = np.fromfile('test2_velocity_spec_i{0:0>5x}'.format(0), dtype = np.float64)
    a.plot(k1, s1)
    a.plot(k2, s2)
    a.set_xscale('log')
    a.set_yscale('log')
    a = fig.add_subplot(122)
    s1 = np.fromfile('test1_velocity_spec_i{0:0>5x}'.format(stats1.shape[0]-1), dtype = np.float64)
    s2 = np.fromfile('test2_velocity_spec_i{0:0>5x}'.format(stats2.shape[0]-1), dtype = np.float64)
    a.plot(k1, s1)
    a.plot(k2, s2)
    a.set_xscale('log')
    a.set_yscale('log')
    fig.savefig('spectra.pdf', format = 'pdf')
    return None

def Kolmogorov_flow_test(opt):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--particles', dest = 'particles', action = 'store_true')
    parser.add_argument('--clean', dest = 'clean', action = 'store_true')
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('--ncpu', type = int, dest = 'ncpu', default = 2)
    parser.add_argument('--nsteps', type = int, dest = 'nsteps', default = 16)
    parser.add_argument('-n', type = int, dest = 'n', default = 64)
    parser.add_argument('--wd', type = str, dest = 'work_dir', default = 'data')
    opt = parser.parse_args()
    #test_curl(opt)
    NStest(opt)
    #resize_test(opt)

