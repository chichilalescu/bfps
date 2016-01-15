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

from base import *

def convergence_test(
        opt,
        code_launch_routine,
        init_vorticity = None,
        code_class = bfps.NavierStokes):
    ### test Navier Stokes convergence
    # first, run code three times, doubling and quadrupling the resolution
    # initial condition and viscosity must be the same!
    opt.simname = 'N{0:0>3x}'.format(opt.n)
    c0 = code_launch_routine(
            opt,
            vorticity_field = init_vorticity,
            code_class = code_class)
    opt.initialize = False
    double(opt)
    opt.n *= 2
    opt.niter_todo *= 2
    opt.niter_stat *= 2
    opt.niter_part *= 2
    opt.ncpu *= 2
    opt.simname = 'N{0:0>3x}'.format(opt.n)
    c1 = code_launch_routine(
            opt,
            nu = c0.parameters['nu'],
            vorticity_field = init_vorticity,
            code_class = code_class,
            tracer_state_file = h5py.File(os.path.join(c0.work_dir, c0.simname + '.h5'), 'r'))
    double(opt)
    opt.n *= 2
    opt.niter_todo *= 2
    opt.niter_stat *= 2
    opt.niter_part *= 2
    opt.ncpu *= 2
    opt.simname = 'N{0:0>3x}'.format(opt.n)
    c2 = code_launch_routine(
            opt,
            nu = c0.parameters['nu'],
            vorticity_field = init_vorticity,
            code_class = code_class,
            tracer_state_file = h5py.File(os.path.join(c0.work_dir, c0.simname + '.h5'), 'r'))
    # get real space fields
    converter = bfps.fluid_converter(
            fluid_precision = opt.precision,
            use_fftw_wisdom = False)
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
        acceleration_test(c)
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
            dashes = (3,3),
            color = (0, 0, 0))
    a.plot( [c0.parameters['dt'], c1.parameters['dt']],
            [c0.parameters['dt']**2, c1.parameters['dt']**2],
            label = '$\\Delta t^2$',
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
    return c0, c1, c2

if __name__ == '__main__':
    opt = parser.parse_args(
            ['-n', '16',
             '--run',
             '--initialize',
             '--ncpu', '2',
             '--nparticles', '1000',
             '--niter_todo', '32',
             '--precision', 'single',
             '--wd', 'data/single'] +
            sys.argv[1:])
    convergence_test(opt, launch)

