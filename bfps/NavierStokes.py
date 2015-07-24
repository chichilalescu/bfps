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

import bfps
import bfps.code
import bfps.tools
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class NavierStokes(bfps.code):
    def __init__(
            self,
            name = 'NavierStokes',
            work_dir = './',
            simname = 'test'):
        super(NavierStokes, self).__init__()
        self.work_dir = work_dir
        self.simname = simname
        self.particle_species = 0
        self.name = name
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['niter_todo'] = 8
        self.parameters['niter_stat'] = 1
        self.parameters['niter_spec'] = 1
        self.parameters['niter_part'] = 1
        self.parameters['dt'] = 0.01
        self.parameters['nu'] = 0.1
        self.parameters['famplitude'] = 1.0
        self.parameters['fmode'] = 1
        self.parameters['fk0'] = 0.0
        self.parameters['fk1'] = 3.0
        self.parameters['forcing_type'] = 'linear'
        self.parameters['nparticles'] = 0
        self.fluid_includes = '#include "fluid_solver.hpp"\n'
        self.fluid_variables = ''
        self.fluid_definitions = ''
        self.fluid_start = ''
        self.fluid_loop = ''
        self.fluid_end  = ''
        self.particle_includes = '#include "tracers.hpp"\n'
        self.particle_variables = ''
        self.particle_definitions = ''
        self.particle_start = ''
        self.particle_loop = ''
        self.particle_end  = ''
        self.fill_up_fluid_code()
        self.style = {}
        return None
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.fluid_variables += ('double stats[4];\n' +
                                 'FILE *stat_file;\n')
        self.fluid_definitions += """
                //begincpp
                void do_stats(fluid_solver<float> *fsolver)
                {
                    if (fsolver->iteration % niter_stat == 0)
                    {
                        double vel_tmp, val_tmp;
                        fsolver->compute_velocity(fsolver->cvorticity);
                        stats[0] = .5*fsolver->correl_vec(fsolver->cvelocity,  fsolver->cvelocity);
                        stats[1] = .5*fsolver->correl_vec(fsolver->cvorticity, fsolver->cvorticity);
                        fsolver->ift_velocity();
                        val_tmp = (fsolver->ru[0]*fsolver->ru[0] +
                                   fsolver->ru[1]*fsolver->ru[1] +
                                   fsolver->ru[2]*fsolver->ru[2]);
                        stats[2] = sqrt(val_tmp);
                        stats[3] = 0.0;
                        RLOOP_FOR_OBJECT(
                            fsolver,
                            val_tmp = (fsolver->ru[rindex*3+0]*fsolver->ru[rindex*3+0] +
                                       fsolver->ru[rindex*3+1]*fsolver->ru[rindex*3+1] +
                                       fsolver->ru[rindex*3+2]*fsolver->ru[rindex*3+2]);
                            stats[3] += val_tmp*.5;
                            vel_tmp = sqrt(val_tmp);
                            if (vel_tmp > stats[2])
                                stats[2] = vel_tmp;
                            );
                        stats[3] /= fsolver->normalization_factor;
                        MPI_Allreduce((void*)(stats + 3), (void*)(&val_tmp), 1, MPI_DOUBLE, MPI_SUM, fsolver->rd->comm);
                        stats[3] = val_tmp;
                        if (myrank == 0)
                        {
                            fwrite((void*)&fsolver->iteration, sizeof(int), 1, stat_file);
                            fwrite((void*)&t, sizeof(double), 1, stat_file);
                            fwrite((void*)stats, sizeof(double), 4, stat_file);
                        }
                    }
                    if (fsolver->iteration % niter_spec == 0)
                    {
                        fsolver->write_spectrum("velocity",  fsolver->cvelocity);
                        fsolver->write_spectrum("vorticity", fsolver->cvorticity);
                    }
                }
                //endcpp
                """
        self.stats_dtype = np.dtype([('iteration', np.int32),
                                     ('t',         np.float64),
                                     ('energy',    np.float64),
                                     ('enstrophy', np.float64),
                                     ('vel_max',   np.float64),
                                     ('renergy',   np.float64)])
        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        pickle.dump(
                self.stats_dtype,
                open(os.path.join(
                        self.work_dir,
                        self.name + '_dtype.pickle'),
                     'w'))
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += ('double t;\n' +
                                 'fluid_solver<float> *fs;\n')
        self.write_fluid_stats()
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs = new fluid_solver<float>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iter0;
                fs->read('v', 'c');
                if (myrank == 0)
                {
                    sprintf(fname, "%s_stats.bin", simname);
                    stat_file = fopen(fname, "ab");
                    sprintf(fname, "%s_time_i%.5x", simname, iter0);
                    FILE *time_file = fopen(fname, "rb");
                    fread((void*)&t, sizeof(double), 1, time_file);
                    fclose(time_file);
                }
                MPI_Bcast((void*)&t, 1, MPI_DOUBLE, 0, fs->cd->comm);
                do_stats(fs);
                //endcpp
                """
        self.fluid_loop += """
                //begincpp
                fs->step(dt);
                t += dt;
                do_stats(fs);
                //endcpp
                """
        self.fluid_end += """
                //begincpp
                if (myrank == 0)
                {
                    fclose(stat_file);
                    sprintf(fname, "%s_time_i%.5x", simname, fs->iteration);
                    FILE *time_file = fopen(fname, "wb");
                    fwrite((void*)&t, sizeof(double), 1, time_file);
                    fclose(time_file);
                }
                fs->write('v', 'c');
                delete fs;
                //endcpp
                """
        return None
    def add_particles(
            self,
            neighbours = 1,
            smoothness = 1,
            kcut = 'fs->kM'):
        self.parameters['neighbours{0}'.format(self.particle_species)] = neighbours
        self.parameters['smoothness{0}'.format(self.particle_species)] = smoothness
        self.parameters['kcut{0}'.format(self.particle_species)] = kcut
        self.particle_variables += 'tracers<float> *ps{0};'.format(self.particle_species)
        self.particle_variables += 'FILE *traj_file{0};\n'.format(self.particle_species)
        #self.particle_definitions
        update_field = ('fs->compute_velocity(fs->cvorticity);\n' +
                        'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut) +
                        'fs->ift_velocity();\n' +
                        'ps{0}->update_field();\n').format(self.particle_species)
        self.particle_start += ('sprintf(fname, "%s_tracers{0}", simname);\n' +
                                'ps{0} = new tracers<float>(\n' +
                                    'fname, fs,\n' +
                                    'nparticles, neighbours{0}, smoothness{0},\n' +
                                    'fs->ru);\n' +
                                'ps{0}->dt = dt;\n' +
                                'ps{0}->iteration = iter0;\n' +
                                update_field +
                                'ps{0}->read();\n' +
                                'if (myrank == 0)\n' +
                                '{{\n' +
                                '    sprintf(fname, "%s_traj.bin", ps{0}->name);\n' +
                                '    traj_file{0} = fopen(fname, "wb");\n' +
                                '    fwrite((void*)(&ps{0}->iteration), sizeof(int), 1, traj_file{0});\n' +
                                '    fwrite((void*)ps{0}->state, sizeof(double), ps{0}->array_size, traj_file{0});\n' +
                                '}}\n').format(self.particle_species)
        self.particle_loop +=  (update_field +
                               'ps{0}->Euler();\n' +
                               'ps{0}->iteration++;\n' +
                               'ps{0}->synchronize();\n' +
                                'if (myrank == 0 && (ps{0}->iteration % niter_part == 0))\n' +
                                '{{\n' +
                                '    fwrite((void*)(&ps{0}->iteration), sizeof(int), 1, traj_file{0});\n' +
                                '    fwrite((void*)ps{0}->state, sizeof(double), ps{0}->array_size, traj_file{0});\n' +
                                '}}\n').format(self.particle_species)
        self.particle_end += ('ps{0}->write();\n' +
                              'if (myrank == 0) fclose(traj_file{0});\n' +
                              'delete ps{0};\n').format(self.particle_species)
        self.particle_species += 1
        return None
    def finalize_code(self):
        self.variables  += self.cdef_pars()
        self.definitions+= self.cread_pars()
        self.includes   += self.fluid_includes
        self.variables  += self.fluid_variables
        self.definitions+= self.fluid_definitions
        if self.particle_species > 0:
            self.includes    += self.particle_includes
            self.variables   += self.particle_variables
            self.definitions += self.particle_definitions
        self.main        = self.fluid_start
        if self.particle_species > 0:
            self.main   += self.particle_start
        self.main       += 'for (; fs->iteration < iter0 + niter_todo;)\n{\n'
        if self.particle_species > 0:
            self.main   += self.particle_loop
        self.main       += self.fluid_loop
        self.main       += '\n}\n'
        if self.particle_species > 0:
            self.main   += self.particle_end
        self.main       += self.fluid_end
        return None
    def read_parameters(
            self,
            simname = None,
            work_dir = None):
        if not type(simname) == type(None):
            self.simname = simname
        if not type(work_dir) == type(None):
            self.work_dir = work_dir
        current_dir = os.getcwd()
        os.chdir(self.work_dir)
        self.read_par(self.simname)
        os.chdir(current_dir)
        return None
    def plot_vel_cut(
            self,
            axis,
            field = 'velocity',
            iteration = 0,
            yval = 13,
            filename = None):
        axis.set_axis_off()
        if type(filename) == type(None):
            filename = self.simname + '_' + field + '_i{0:0>5x}'.format(iteration)
        Rdata0 = np.fromfile(
                filename,
                dtype = np.float32).reshape((-1,
                                             self.parameters['ny'],
                                             self.parameters['nx'], 3))
        energy = np.sum(Rdata0[:, yval, :, :]**2, axis = 2)*.5
        axis.imshow(energy, interpolation='none')
        axis.set_title('{0}'.format(np.average(Rdata0[..., 0]**2 +
                                               Rdata0[..., 1]**2 +
                                               Rdata0[..., 2]**2)*.5))
        return Rdata0
    def generate_vdf(
            self,
            field = 'velocity',
            iteration = 0,
            filename = None):
        if type(filename) == type(None):
            filename = self.simname + '_' + field + '_i{0:0>5x}'.format(iteration)
        Rdata0 = np.fromfile(
                filename,
                dtype = np.float32).reshape((self.parameters['nz'],
                                             self.parameters['ny'],
                                             self.parameters['nx'], 3))
        subprocess.call(['vdfcreate',
                         '-dimension',
                         '{0}x{1}x{2}'.format(self.parameters['nz'],
                                              self.parameters['ny'],
                                              self.parameters['nx']),
                         '-numts', '1',
                         '-varnames', '{0}x:{0}y:{0}z'.format(field),
                         filename + '.vdf'])
        for loop_data in [(0, 'x'), (1, 'y'), (2, 'z')]:
            Rdata0[..., loop_data[0]].tofile('tmprawfile')
            subprocess.call(['raw2vdf',
                             '-ts', '0',
                             '-varname', '{0}{1}'.format(field, loop_data[1]),
                             filename + '.vdf',
                             'tmprawfile'])
        return Rdata0
    def generate_vector_field(
            self,
            rseed = 7547,
            spectra_slope = 1.,
            precision = 'single',
            iteration = 0,
            write_to_file = False):
        if precision == 'single':
            dtype = np.complex64
        elif precision == 'double':
            dtype = np.complex128
        np.random.seed(rseed)
        Kdata00 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata01 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata02 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        Kdata1 = bfps.tools.padd_with_zeros(
                Kdata0,
                self.parameters['nz'],
                self.parameters['ny'],
                self.parameters['nx'])
        if write_to_file:
            Kdata1.tofile(
                    os.path.join(self.work_dir,
                                 self.simname + "_cvorticity_i{0:0>5x}".format(iteration)))
        return Kdata1
    def generate_tracer_state(
            self,
            rseed = 34982,
            iteration = 0,
            species = 0,
            write_to_file = False):
        np.random.seed(rseed*self.particle_species + species)
        data = np.random.random(self.parameters['nparticles']*3)*2*np.pi
        if write_to_file:
            data.tofile(
                    os.path.join(
                        self.work_dir,
                        self.simname + "_tracers{0}_state_i{1:0>5x}".format(species, iteration)))
        return data
    def read_spec(
            self,
            field = 'velocity'):
        k = np.fromfile(
                os.path.join(
                    self.work_dir,
                    self.simname + '_kshell'),
                dtype = np.float64)
        spec_dtype = np.dtype([('iteration', np.int32),
                               ('val', np.float64, k.shape[0])])
        spec = np.fromfile(
                os.path.join(
                    self.work_dir,
                    self.simname + '_' + field + '_spec'),
                dtype = spec_dtype)
        return k, spec
    def read_stats(self):
        dtype = pickle.load(open(
                os.path.join(self.work_dir, self.name + '_dtype.pickle'), 'r'))
        return np.fromfile(os.path.join(self.work_dir, self.simname + '_stats.bin'),
                           dtype = dtype)
    def read_traj(self):
        if self.particle_species == 0:
            return None
        pdtype = np.dtype([('iteration', np.int32),
                           ('state', np.float64, (self.parameters['nparticles'], 3))])
        traj_list = []
        for t in range(self.particle_species):
            traj_list.append(np.fromfile(
                    os.path.join(
                        self.work_dir,
                        self.simname + '_tracers{0}_traj.bin'.format(t)),
                    dtype = pdtype))
        traj = np.zeros((self.particle_species, traj_list[0].shape[0]), dtype = pdtype)
        for t in range(self.particle_species):
            traj[t] = traj_list[t]
        return traj
    def generate_initial_condition(self):
        np.array([0.0]).tofile(
                os.path.join(
                        self.work_dir, self.simname + '_time_i00000'))
        self.generate_vector_field(write_to_file = True)
        for species in range(self.particle_species):
            self.generate_tracer_state(
                    species = species,
                    write_to_file = True)
        return None
    def basic_plots(
            self,
            spectra_on = True,
            particles_on = True):
        k = np.fromfile(
                os.path.join(
                    self.work_dir,
                    self.simname + '_kshell'),
                dtype = np.float64)

        # plot energy and enstrophy
        stats = self.read_stats()
        fig = plt.figure(figsize = (12, 6))
        a = fig.add_subplot(121)
        etaK = (self.parameters['nu']**2 / (stats['enstrophy']*2))**.25
        a.plot(stats['t'], k[-3]*etaK, label = '$k_M \eta_K$')
        a.plot(stats['t'], self.parameters['dt']*stats['vel_max'] / (2*np.pi/self.parameters['nx']),
                label = '$\\frac{\\Delta t \\| u \\|_\infty}{\\Delta x}$')
        a.legend(loc = 'best')
        a = fig.add_subplot(122)
        a.plot(stats['t'], stats['energy'], label = 'energy', color = (0, 0, 1))
        a.plot(stats['t'], stats['renergy'], label = 'energy', color = (1, 0, 1), dashes = (2, 2))
        a.set_ylabel('energy', color = (0, 0, 1))
        a.set_xlabel('$t$')
        for tt in a.get_yticklabels():
            tt.set_color((0, 0, 1))
        b = a.twinx()
        b.plot(stats['t'], stats['enstrophy'], label = 'enstrophy', color = (1, 0, 0))
        b.set_ylabel('enstrophy', color = (1, 0, 0))
        for tt in b.get_yticklabels():
            tt.set_color((1, 0, 0))
        fig.savefig('stats.pdf', format = 'pdf')

        # plot spectra
        if spectra_on:
            fig = plt.figure(figsize=(12, 6))
            a = fig.add_subplot(121)
            self.plot_spectrum(a, average = False)
            a.set_title('velocity')
            a = fig.add_subplot(122)
            self.plot_spectrum(a, average = False, field = 'vorticity')
            a.set_title('vorticity')
            fig.savefig('spectrum.pdf', format = 'pdf')


        # plot particles
        if particles_on and self.particle_species > 0:
            traj = self.read_traj()
            fig = plt.figure(figsize = (12, 12))
            a = fig.add_subplot(111, projection = '3d')
            for t in range(self.parameters['nparticles']):
                a.plot(traj['state'][0, :, t, 0],
                       traj['state'][0, :, t, 1],
                       traj['state'][0, :, t, 2], color = 'blue')
                a.plot(traj['state'][1, :, t, 0],
                       traj['state'][1, :, t, 1],
                       traj['state'][1, :, t, 2], color = 'red', dashes = (1, 1))
            fig.savefig('traj.pdf', format = 'pdf')
        return None
    def compute_statistics(self, iter0 = 0):
        self.read_parameters()
        stats = self.read_stats()
        assert(stats.shape[0] > 0)
        iter0 = min(stats['iteration'][-1], iter0)
        index0 = np.where(stats['iteration'] == iter0)[0][0]
        self.statistics = {}
        self.statistics['t'] = stats['t'][index0:]
        self.statistics['t_indices'] = stats['iteration'][index0:]
        for key in ['energy', 'enstrophy', 'vel_max']:
            self.statistics[key + '(t)'] = stats[key][index0:]
            self.statistics[key] = np.average(stats[key][index0:])
        for suffix in ['', '(t)']:
            self.statistics['diss'    + suffix] = (self.parameters['nu'] *
                                                   self.statistics['enstrophy' + suffix]*2)
            self.statistics['etaK'    + suffix] = (self.parameters['nu']**3 /
                                                   self.statistics['diss' + suffix])**.25
            self.statistics['Rlambda' + suffix] = (2*np.sqrt(5./3) *
                                                   (self.statistics['energy' + suffix] /
                                                   (self.parameters['nu']*self.statistics['diss' + suffix])**.5))
            self.statistics['tauK'    + suffix] =  (self.parameters['nu'] /
                                                    self.statistics['diss' + suffix])**.5
        k, spec = self.read_spec(field = 'velocity')
        assert(spec.shape[0] > 0 and iter0 < spec['iteration'][-1])
        self.statistics['k'] = k
        self.statistics['kM'] = np.nanmax(k)
        index0 = np.where(spec['iteration'] == iter0)[0][0]
        self.statistics['spec_indices'] = spec['iteration'][index0:]
        list_of_indices = []
        for bla in self.statistics['spec_indices']:
            list_of_indices.append(np.where(self.statistics['t_indices'] == bla)[0][0])
        self.statistics['spec_t'] = self.statistics['t'][list_of_indices]
        self.statistics['energy(t, k)'] = spec[index0:]['val'] / 2
        self.statistics['energy(k)'] = np.average(spec[index0:]['val'], axis = 0) / 2
        self.trajectories = self.read_traj()
        #if self.particle_species > 0: get some velocity histograms or smth
        return None
    def plot_spectrum(
            self,
            axis,
            iter0 = 0,
            field = 'velocity',
            average = True,
            color = (1, 0, 0),
            cmap = 'coolwarm',
            add_Kspec = True,
            normalize_k = True,
            normalization = 'energy',
            label = None):
        self.compute_statistics()
        stats = self.read_stats()
        k, spec = self.read_spec(field = field)
        index = np.where(spec['iteration'] == iter0)[0][0]
        sindex = np.where(stats['iteration'] == iter0)[0][0]
        E = np.average(stats['energy'][sindex:])
        aens = np.average(stats['enstrophy'][sindex:])
        diss = self.parameters['nu']*aens*2
        etaK = (self.parameters['nu']**2 / (aens*2))**.25
        norm_factor = 1.0
        if normalization == 'energy':
            norm_factor = (self.parameters['nu']**5 * diss)**(-.25)
        if normalize_k:
            k *= etaK
        if average:
            aspec = np.average(spec[index:]['val'], axis = 0)
            axis.plot(
                    k,
                    aspec*norm_factor,
                    color = color,
                    label = label)
        else:
            for i in range(index, spec.shape[0]):
                axis.plot(k,
                          spec[i]['val']*norm_factor,
                          color = plt.get_cmap(cmap)((i - iter0)*1.0/(spec.shape[0] - iter0)))
        if add_Kspec:
            axis.plot(
                    k,
                    2*k**(-5./3),
                    color = 'black',
                    dashes = (1, 1),
                    label = '$2(k \\eta_K)^{-5/3}$')
        axis.set_xscale('log')
        axis.set_yscale('log')
        return k, spec
    def set_plt_style(
            self,
            style = {'dashes' : (None, None)}):
        self.style.update(style)
        return None

import subprocess

def launch(
        opt,
        nu = None):
    c = NavierStokes(work_dir = opt.work_dir)
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
    c.parameters['niter_stat'] = 1
    c.parameters['niter_spec'] = 4
    c.parameters['niter_part'] = 2
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = 32
    if opt.particles:
        c.add_particles()
        c.add_particles(kcut = 'fs->kM/2')
    c.finalize_code()
    c.write_src()
    c.write_par(simname = c.simname)
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            c.generate_initial_condition()
        for nrun in range(opt.njobs):
            c.run(ncpu = opt.ncpu,
                  simname = 'test',
                  iter0 = opt.iteration + nrun*opt.nsteps)
    return c

def test(opt):
    c = launch(opt)
    c.basic_plots()
    return None

