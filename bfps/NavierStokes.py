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
import bfps.fluid_base
import bfps.tools
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class NavierStokes(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'NavierStokes',
            work_dir = './',
            simname = 'test'):
        super(NavierStokes, self).__init__(name = name, work_dir = work_dir, simname = simname)
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
                if (fs->iteration % niter_out == 0)
                    fs->write('v', 'c');
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
                if (fs->iteration % niter_out != 0)
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

