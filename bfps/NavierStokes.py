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

class NavierStokes(bfps.code):
    def __init__(
            self,
            name = 'NavierStokes',
            work_dir = './'):
        super(NavierStokes, self).__init__()
        self.work_dir = work_dir
        self.particle_species = 0
        self.name = name
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['niter_todo'] = 8
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
        return None
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_variables += ('double stats[3];\n' +
                                 'FILE *stat_file;\n')
        self.fluid_definitions += """
                //begincpp
                void do_stats(fluid_solver<float> *fsolver)
                {
                    double vel_tmp;
                    fsolver->compute_velocity(fsolver->cvorticity);
                    stats[0] = .5*fsolver->correl_vec(fsolver->cvelocity,  fsolver->cvelocity);
                    stats[1] = .5*fsolver->correl_vec(fsolver->cvorticity, fsolver->cvorticity);
                    fs->ift_velocity();
                    stats[2] = sqrt(fs->ru[0]*fs->ru[0] +
                                    fs->ru[1]*fs->ru[1] +
                                    fs->ru[2]*fs->ru[2]);
                    for (ptrdiff_t rindex = 1; rindex < fs->rd->local_size; rindex++)
                    {
                        vel_tmp = sqrt(fs->ru[rindex*3+0]*fs->ru[rindex*3+0] +
                                       fs->ru[rindex*3+1]*fs->ru[rindex*3+1] +
                                       fs->ru[rindex*3+2]*fs->ru[rindex*3+2]);
                        if (vel_tmp > stats[2])
                            stats[2] = vel_tmp;
                    }
                    if (myrank == 0)
                    {
                        fwrite((void*)&fsolver->iteration, sizeof(int), 1, stat_file);
                        fwrite((void*)&t, sizeof(double), 1, stat_file);
                        fwrite((void*)stats, sizeof(double), 3, stat_file);
                    }
                    fs->write_spectrum("velocity", fs->cvelocity);
                    fs->write_spectrum("vorticity", fs->cvorticity);
                }
                //endcpp
                """
        self.stats_dtype = np.dtype([('iteration', np.int32),
                                     ('t',         np.float64),
                                     ('energy',    np.float64),
                                     ('enstrophy', np.float64),
                                     ('vel_max',   np.float64)])
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
                }
                t = dt*iter0;
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
                }
                fs->write('v', 'c');
                fs->write('u', 'r');
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
                                '    fwrite((void*)ps{0}->state, sizeof(double), ps{0}->array_size, traj_file{0});\n' +
                                '}}\n').format(self.particle_species)
        self.particle_loop +=  (update_field +
                               'ps{0}->Euler();\n' +
                               'ps{0}->iteration++;\n' +
                               'ps{0}->synchronize();\n').format(self.particle_species)
        self.particle_end += ('ps{0}->write();\n' +
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
    def plot_vel_cut(
            self,
            axis,
            simname = 'test',
            field = 'velocity',
            iteration = 0,
            yval = 13,
            filename = None):
        axis.set_axis_off()
        if type(filename) == type(None):
            filename = simname + '_' + field + '_i{0:0>5x}'.format(iteration)
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
    def generate_vector_field(
            self,
            rseed = 7547,
            spectra_slope = 1.,
            precision = 'single',
            simname = None,
            iteration = 0):
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
        if not (type(simname) == type(None)):
            Kdata1.tofile(
                    os.path.join(self.work_dir,
                                 simname + "_cvorticity_i{0:0>5x}".format(iteration)))
        return Kdata1
    def generate_tracer_state(
            self,
            rseed = 34982,
            simname = None,
            iteration = 0,
            species = 0):
        np.random.seed(rseed*self.particle_species + species)
        data = np.random.random(self.parameters['nparticles']*3)*2*np.pi
        if not (type(simname) == type(None)):
            data.tofile(
                    os.path.join(
                        self.work_dir,
                        simname + "_tracers{0}_state_i{1:0>5x}".format(species, iteration)))
        return data
    def read_spec(
            self,
            simname = 'test',
            field = 'velocity'):
        k = np.fromfile(
                os.path.join(
                    self.work_dir,
                    simname + '_kshell'),
                dtype = np.float64)
        spec_dtype = np.dtype([('iteration', np.int32),
                               ('val', np.float64, k.shape[0])])
        spec = np.fromfile(
                os.path.join(
                    self.work_dir,
                    simname + '_' + field + '_spec'),
                dtype = spec_dtype)
        return k, spec
    def read_stats(
            self, simname = 'test'):
        dtype = pickle.load(open(
                os.path.join(self.work_dir, self.name + '_dtype.pickle'), 'r'))
        return np.fromfile(os.path.join(self.work_dir, simname + '_stats.bin'),
                           dtype = dtype)

import subprocess
import matplotlib.pyplot as plt

def test(opt):
    if opt.run or opt.clean:
        subprocess.call(['rm {0}/test_*'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*.pickle'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*.elf'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*version_info.txt'.format(opt.work_dir)], shell = True)
    if opt.clean:
        subprocess.call(['make', 'clean'])
        return None
    c = NavierStokes(work_dir = opt.work_dir)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 5.5*opt.n**(-4./3)
    c.parameters['dt'] = 5e-3
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['famplitude'] = 1.
    c.parameters['nparticles'] = 32
    if opt.particles:
        c.add_particles()
        c.add_particles(kcut = 'fs->kM/2')
    c.finalize_code()
    c.write_src()
    c.write_par(simname = 'test')
    if opt.run:
        c.generate_vector_field(simname = 'test')
        if opt.particles:
            c.generate_tracer_state(simname = 'test', species = 0)
            c.generate_tracer_state(simname = 'test', species = 1)
        c.run(ncpu = opt.ncpu,
              simname = 'test')
    stats = c.read_stats()
    k, enespec = c.read_spec()
    k, ensspec = c.read_spec(field = 'vorticity')

    # plot energy and enstrophy
    fig = plt.figure(figsize = (12, 6))
    a = fig.add_subplot(121)
    etaK = (c.parameters['nu']**2 / (stats['enstrophy']*2))**.25
    a.plot(stats['t'], k[-3]*etaK, label = '$k_M \eta_K$')
    a.plot(stats['t'], c.parameters['dt']*stats['vel_max'] / (2*np.pi/c.parameters['nx']),
            label = '$\\frac{\\Delta t \\| u \\|_\infty}{\\Delta x}$')
    a.legend(loc = 'best')
    a = fig.add_subplot(122)
    a.plot(stats['t'], stats['energy'], label = 'energy', color = (0, 0, 1))
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
    fig = plt.figure(figsize=(12,6))
    a = fig.add_subplot(121)
    for i in range(0, enespec.shape[0]):
        a.plot(k, enespec[i]['val'], color = (i*1./enespec.shape[0], 0, 1 - i*1./enespec.shape[0]))
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_title('velocity')
    a = fig.add_subplot(122)
    for i in range(0, ensspec.shape[0], 64):
        a.plot(k, ensspec[i]['val'],
               color = (i*1./ensspec.shape[0], 0, 1 - i*1./ensspec.shape[0]))
        a.plot(k, k**2*enespec[i]['val'],
               color = (0, 1 - i*1./ensspec.shape[0], i*1./ensspec.shape[0]),
               dashes = (2, 2))
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_title('vorticity')
    fig.savefig('spectrum.pdf', format = 'pdf')
    return None

