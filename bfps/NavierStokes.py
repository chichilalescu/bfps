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
import numpy as np
import h5py
import matplotlib.pyplot as plt

import bfps
import bfps.fluid_base
import bfps.tools

class NavierStokes(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'NavierStokes',
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single'):
        super(NavierStokes, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision)
        self.file_datasets_create = """
                //begincpp
                std::string temp_string;
                H5::DataSet dset;
                H5::Group group;
                H5::DataSpace tkfunc_dspace;
                H5::DataSpace tmp_dspace;
                H5::DataSpace tfunc_dspace;
                H5::DataSpace kfunc_dspace;
                H5::DataSpace value_dspace;
                H5::DataSpace kcomp_dspace;
                hsize_t dims[4];
                hsize_t maxdims[4];
                H5::DSetCreatPropList cparms;
                double fill_val = 0;
                cparms.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_val);
                H5::FloatType double_dtype(H5::PredType::NATIVE_DOUBLE);
                H5::IntType ptrdiff_t_dtype(H5::PredType::NATIVE_INT64); //is this ok?

                // first, generate the various data spaces that are to be used
                dims[0] = niter_todo+1;
                dims[1] = fs->nshells;
                maxdims[0] = H5S_UNLIMITED;
                maxdims[1] = dims[1];
                tfunc_dspace  = H5::DataSpace(1, dims, maxdims);
                tkfunc_dspace = H5::DataSpace(2, dims, maxdims);
                dims[0] = fs->nshells;
                kfunc_dspace  = H5::DataSpace(1, dims);
                dims[0] = 1;
                value_dspace   = H5::DataSpace(1, dims);

                // store the information about kspace
                group = data_file.createGroup("/kspace");
                group.createDataSet("kshell", double_dtype, kfunc_dspace);
                group.createDataSet("nshell", ptrdiff_t_dtype, kfunc_dspace);
                group.createDataSet("kM", double_dtype, value_dspace);
                group.createDataSet("dk", double_dtype, value_dspace);
                dset = data_file.openDataSet("/kspace/kshell");
                dset.write(fs->kshell, double_dtype);
                dset = data_file.openDataSet("/kspace/nshell");
                dset.write(fs->nshell, ptrdiff_t_dtype);
                dset = data_file.openDataSet("/kspace/kM");
                dset.write(&fs->kM, double_dtype);
                dset = data_file.openDataSet("/kspace/dk");
                dset.write(&fs->dk, double_dtype);
                dims[0] = fs->cd->sizes[2];
                kcomp_dspace = H5::DataSpace(1, dims);
                group.createDataSet("kx", double_dtype, kcomp_dspace);
                dset = data_file.openDataSet("/kspace/kx");
                dset.write(fs->kx, double_dtype);
                dims[0] = fs->cd->sizes[1];
                kcomp_dspace = H5::DataSpace(1, dims);
                group.createDataSet("kz", double_dtype, kcomp_dspace);
                dset = data_file.openDataSet("/kspace/kz");
                dset.write(fs->kz, double_dtype);
                dims[0] = fs->cd->sizes[0];
                kcomp_dspace = H5::DataSpace(1, dims);
                group.createDataSet("ky", double_dtype, kcomp_dspace);
                dset = data_file.openDataSet("/kspace/ky");
                double *tmp_ky = new double[fs->cd->sizes[0]];
                for (int i = 0; i<fs->cd->sizes[0]; i++)
                    tmp_ky[i] = ((i + fs->cd->sizes[0]/2-1)%fs->cd->sizes[0] - fs->cd->sizes[0]/2+1)*fs->dky;
                dset.write(tmp_ky, double_dtype);
                delete[] tmp_ky;

                // generate datasets for various statistics
                group = data_file.openGroup("/statistics");
                hsize_t chunk_dims[4];
                chunk_dims[0] = 1;
                chunk_dims[1] = dims[1];
                cparms.setChunk(1, chunk_dims);
                group.createDataSet("maximum_velocity"  , double_dtype, tfunc_dspace , cparms);
                group.createDataSet("realspace_energy"  , double_dtype, tfunc_dspace , cparms);
                cparms.setChunk(2, chunk_dims);
                group.createDataSet("spectrum_velocity" , double_dtype, tkfunc_dspace, cparms);
                group.createDataSet("spectrum_vorticity", double_dtype, tkfunc_dspace, cparms);
                //endcpp
                """
        self.file_datasets_grow = """
                //begincpp
                H5::DataSet dset;
                H5::DataSpace dspace;
                std::string temp_string;
                hsize_t dims[4];
                hsize_t old_dims[4];
                dset = data_file.openDataSet("/statistics/maximum_velocity");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(old_dims);
                dims[0] = niter_todo + old_dims[0];
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/realspace_energy");
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/spectrum_velocity");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(old_dims);
                dims[0] = niter_todo + old_dims[0];
                dims[1] = old_dims[1];
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/spectrum_vorticity");
                dset.extend(dims);
                //endcpp
                """
        self.style = {}
        self.statistics = {}
        return None
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.stat_src += """
                //begincpp
                    double vel_tmp, val_tmp;
                    double max_vel, rspace_energy;
                    fs->compute_velocity(fs->cvorticity);
                    fs->ift_velocity();
                    val_tmp = (fs->ru[0]*fs->ru[0] +
                               fs->ru[1]*fs->ru[1] +
                               fs->ru[2]*fs->ru[2]);
                    max_vel = sqrt(val_tmp);
                    rspace_energy = 0.0;
                    RLOOP_FOR_OBJECT(
                        fs,
                        val_tmp = (fs->ru[rindex*3+0]*fs->ru[rindex*3+0] +
                                   fs->ru[rindex*3+1]*fs->ru[rindex*3+1] +
                                   fs->ru[rindex*3+2]*fs->ru[rindex*3+2]);
                        rspace_energy += val_tmp*.5;
                        vel_tmp = sqrt(val_tmp);
                        if (vel_tmp > max_vel)
                            max_vel = vel_tmp;
                        );
                    rspace_energy /= fs->normalization_factor;
                    MPI_Allreduce((void*)(&rspace_energy), (void*)(&val_tmp), 1, MPI_DOUBLE, MPI_SUM, fs->rd->comm);
                    rspace_energy = val_tmp;
                    MPI_Allreduce((void*)(&max_vel), (void*)(&val_tmp), 1, MPI_DOUBLE, MPI_MAX, fs->rd->comm);
                    max_vel = val_tmp;
                    double *spec_velocity = fftw_alloc_real(fs->nshells);
                    double *spec_vorticity = fftw_alloc_real(fs->nshells);
                    fs->cospectrum(fs->cvelocity, fs->cvelocity, spec_velocity);
                    fs->cospectrum(fs->cvorticity, fs->cvorticity, spec_vorticity);
                    if (myrank == 0)
                    {
                        H5::DataSet dset;
                        H5::DataSpace memspace, writespace;
                        hsize_t count[2], offset[2];
                        dset = data_file.openDataSet("statistics/maximum_velocity");
                        writespace = dset.getSpace();
                        count[0] = 1;
                        offset[0] = fs->iteration;
                        memspace = H5::DataSpace(1, count);
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(&max_vel, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        dset = data_file.openDataSet("statistics/realspace_energy");
                        writespace = dset.getSpace();
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(&rspace_energy, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        dset = data_file.openDataSet("statistics/spectrum_velocity");
                        writespace = dset.getSpace();
                        count[1] = fs->nshells;
                        memspace = H5::DataSpace(2, count);
                        offset[1] = 0;
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(spec_velocity, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        dset = data_file.openDataSet("statistics/spectrum_vorticity");
                        writespace = dset.getSpace();
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(spec_vorticity, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                    }
                    fftw_free(spec_velocity);
                    fftw_free(spec_vorticity);
                //endcpp
                """
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += 'fluid_solver<{0}> *fs;\n'.format(self.C_dtype)
        self.write_fluid_stats()
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs = new fluid_solver<{0}>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iteration;
                fs->read('v', 'c');
                //endcpp
                """.format(self.C_dtype)
        self.fluid_loop += """
                //begincpp
                fs->step(dt);
                if (fs->iteration % niter_out == 0)
                    fs->write('v', 'c');
                //endcpp
                """
        self.fluid_end += """
                //begincpp
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
            integration_steps = 2,
            kcut = 'fs->kM'):
        self.parameters['neighbours{0}'.format(self.particle_species)] = neighbours
        self.parameters['smoothness{0}'.format(self.particle_species)] = smoothness
        self.parameters['kcut{0}'.format(self.particle_species)] = kcut
        self.parameters['integration_steps{0}'.format(self.particle_species)] = integration_steps
        self.particle_variables += 'tracers<{0}> *ps{1};'.format(self.C_dtype, self.particle_species)
        self.file_datasets_create += ('tmp_dspace = H5::DataSpace(2, dims, maxdims);\n' +
                                      'temp_string = std::string("/particles/") + std::string(ps{0}->name);\n' +
                                      'group = data_file.openGroup(temp_string);\n' +
                                      'dims[0] = 1;\n' +
                                      'dims[1] = integration_steps{0};\n' +
                                      'dims[2] = nparticles;\n' +
                                      'dims[3] = ps{0}->ncomponents;\n' +
                                      'maxdims[0] = H5S_UNLIMITED;\n' +
                                      'maxdims[1] = dims[1];\n' +
                                      'maxdims[2] = dims[2];\n' +
                                      'maxdims[3] = dims[3];\n' +
                                      'tmp_dspace = H5::DataSpace(4, dims, maxdims);\n' +
                                      'chunk_dims[1] = 1;\n' +
                                      'chunk_dims[1] = 1;\n' +
                                      'chunk_dims[2] = dims[2];\n' +
                                      'chunk_dims[3] = dims[3];\n' +
                                      'cparms.setChunk(4, chunk_dims);\n' +
                                      'group.createDataSet("rhs", double_dtype, tmp_dspace, cparms);\n').format(self.particle_species)
        grow_template = ('temp_string = std::string("/particles/") + std::string(ps{0}->name) + std::string("/{1}");\n' +
                         'dset = data_file.openDataSet(temp_string);\n' +
                         'dspace = dset.getSpace();\n' +
                         'dspace.getSimpleExtentDims(dims);\n' +
                         'dims[0] += {2};\n' +
                         'dset.extend(dims);\n')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'state', '(niter_todo/niter_part)')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'rhs', '1')
        #self.particle_definitions
        update_field = ('fs->compute_velocity(fs->cvorticity);\n' +
                        'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut) +
                        'fs->ift_velocity();\n' +
                        'ps{0}->update_field();\n').format(self.particle_species)
        self.particle_start += ('sprintf(fname, "tracers{1}");\n' +
                                'ps{1} = new tracers<{0}>(\n' +
                                    'fname, fs,\n' +
                                    'nparticles,\n' +
                                    'neighbours{1}, smoothness{1}, niter_part, integration_steps{1},\n' +
                                    'fs->ru);\n' +
                                'ps{1}->dt = dt;\n' +
                                'ps{1}->iteration = iteration;\n' +
                                update_field +
                                'ps{1}->read(&data_file);\n').format(self.C_dtype, self.particle_species)
        self.particle_loop +=  (update_field +
                               'ps{0}->step();\n' +
                                'if (ps{0}->iteration % niter_part == 0)\n' +
                                'ps{0}->write(&data_file, false);\n').format(self.particle_species)
        self.particle_end += ('ps{0}->write(&data_file);\n' +
                              'delete ps{0};\n').format(self.particle_species)
        self.particle_species += 1
        return None
    def basic_plots(
            self,
            spectra_on = True,
            particles_on = True):
        self.compute_statistics()

        # plot energy and enstrophy
        fig = plt.figure(figsize = (12, 6))
        a = fig.add_subplot(121)
        a.plot(self.statistics['t'], self.statistics['kM']*self.statistics['etaK(t)'], label = '$k_M \eta_K$')
        a.plot(self.statistics['t'],
               (self.parameters['dt']*self.statistics['vel_max(t)'] /
                (2*np.pi/self.parameters['nx'])),
               label = '$\\frac{\\Delta t}{\\Delta x} \\| u \\|_\infty$')
        a.legend(loc = 'best')
        a = fig.add_subplot(122)
        a.plot(self.statistics['t'], self.statistics['energy(t)'], label = 'energy', color = (0, 0, 1))
        a.plot(self.statistics['t'], self.statistics['renergy(t)'], label = 'energy', color = (0, 0, 1))
        a.set_ylabel('energy', color = (0, 0, 1))
        a.set_xlabel('$t$')
        for tt in a.get_yticklabels():
            tt.set_color((0, 0, 1))
        b = a.twinx()
        b.plot(self.statistics['t'], self.statistics['enstrophy(t)'], label = 'enstrophy', color = (1, 0, 0))
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
            fig = plt.figure(figsize = (12, 12))
            a = fig.add_subplot(111, projection = '3d')
            for t in range(self.parameters['nparticles']):
                a.plot(self.trajectories[0][:, t, 0],
                       self.trajectories[0][:, t, 1],
                       self.trajectories[0][:, t, 2], color = 'blue')
                a.plot(self.trajectories[1][:, t, 0],
                       self.trajectories[1][:, t, 1],
                       self.trajectories[1][:, t, 2], color = 'red', dashes = (1, 1))
            fig.savefig('traj.pdf', format = 'pdf')
        return None
    def compute_statistics(self, iter0 = 0):
        self.read_parameters()
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r') as data_file:
            iter0 = min(data_file['statistics/maximum_velocity'].shape[0]-1, iter0)
            iter1 = data_file['statistics/maximum_velocity'].shape[0]
            self.statistics['t'] = self.parameters['dt']*np.arange(iter0, iter1).astype(np.float)
            self.statistics['energy(t, k)'] = data_file['statistics/spectrum_velocity'].value/2
            self.statistics['enstrophy(t, k)'] = data_file['statistics/spectrum_vorticity'].value/2
            self.statistics['vel_max(t)'] = data_file['statistics/maximum_velocity'].value
            self.statistics['renegergy(t)'] = data_file['statistics/realspace_energy'].value
            for key in ['energy', 'enstrophy']:
                self.statistics[key + '(t)'] = np.sum(self.statistics[key + '(t, k)'], axis = 1)
            for key in ['energy', 'enstrophy', 'vel_max']:
                self.statistics[key] = np.average(self.statistics[key + '(t)'], axis = 0)
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
            self.statistics['kshell'] = data_file['kspace/kshell'].value
            self.statistics['kM'] = data_file['kspace/kM'].value
            if self.particle_species > 0:
                self.trajectories = [data_file['particles/' + key + '/state'].value
                                     for key in data_file['particles'].keys()]
        return None
    def plot_spectrum(
            self,
            axis,
            quantity = 'energy',
            average = True,
            color = (1, 0, 0),
            cmap = 'coolwarm',
            add_Kspec = True,
            normalize_k = True,
            normalization = 'energy',
            label = None):
        self.compute_statistics()
        norm_factor = 1.0
        if normalization == 'energy':
            norm_factor = (self.parameters['nu']**5 * self.statistics['diss'])**(-.25)
        k = self.statistics['kshell'].copy()
        if normalize_k:
            k *= self.statistics['etaK']
        if average:
            axis.plot(
                    k,
                    self.statistics[quantity + '(k)']*norm_factor,
                    color = color,
                    label = label)
        else:
            for i in range(self.statistics[quantity + '(t, k)'].shape[0]):
                axis.plot(k,
                          self.statistics[quantity + '(t, k)'][i]*norm_factor,
                          color = plt.get_cmap(cmap)(i*1.0/self.statistics[quantity + '(t, k)'].shape[0]))
        if add_Kspec:
            axis.plot(
                    k,
                    2*k**(-5./3),
                    color = 'black',
                    dashes = (1, 1),
                    label = '$2(k \\eta_K)^{-5/3}$')
        axis.set_xscale('log')
        axis.set_yscale('log')
        return None
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
    c.parameters['niter_part'] = 2
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = 32
    if opt.particles:
        c.add_particles()
        c.add_particles(kcut = 'fs->kM/2')
    c.finalize_code()
    c.write_src()
    c.write_par()
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            c.generate_initial_condition()
        c.run(ncpu = opt.ncpu, njobs = opt.njobs)
    return c

def test(opt):
    c = launch(opt)
    c.basic_plots()
    return None

