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
        self.file_datasets_grow = """
                //begincpp
                H5::IntType ptrdiff_t_dtype(H5::PredType::NATIVE_INT64); //is this ok?
                H5::FloatType double_dtype(H5::PredType::NATIVE_DOUBLE);
                H5::DataSet dset;
                H5::DataSpace dspace;
                H5::DSetCreatPropList cparms;
                std::string temp_string;
                hsize_t dims[4];
                hsize_t old_dims[4];
                hsize_t chunk_dims[4];
                hsize_t tdim;
                // store kspace information
                dset = data_file.openDataSet("/kspace/kshell");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(dims);
                if (fs->nshells != dims[0])
                {
                    std::cerr << "ERROR: computed nshells not equal to data file nshells\\n" << std::endl;
                    file_problems++;
                }
                dset.write(fs->kshell, double_dtype);
                dset = data_file.openDataSet("/kspace/nshell");
                dset.write(fs->nshell, ptrdiff_t_dtype);
                dset = data_file.openDataSet("/kspace/kM");
                dset.write(&fs->kM, double_dtype);
                dset = data_file.openDataSet("/kspace/dk");
                dset.write(&fs->dk, double_dtype);
                // moments
                dset = data_file.openDataSet("/statistics/moments/velocity");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(old_dims);
                cparms = dset.getCreatePlist();
                cparms.getChunk(3, chunk_dims);
                tdim = (fs->iteration + niter_todo) - old_dims[0];
                if (tdim > 0)
                    tdim = tdim + (chunk_dims[0] - tdim % chunk_dims[0]) % chunk_dims[0];
                else
                    tdim = 0;
                dims[0] = tdim + old_dims[0];
                dims[1] = old_dims[1];
                dims[2] = old_dims[2];
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/moments/vorticity");
                dset.extend(dims);
                // histograms
                dset = data_file.openDataSet("/statistics/histograms/velocity");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(old_dims);
                cparms = dset.getCreatePlist();
                cparms.getChunk(3, chunk_dims);
                tdim = (fs->iteration + niter_todo) - old_dims[0];
                if (tdim > 0)
                    tdim = tdim + (chunk_dims[0] - tdim % chunk_dims[0]) % chunk_dims[0];
                else
                    tdim = 0;
                dims[0] = tdim + old_dims[0];
                dims[1] = old_dims[1];
                dims[2] = old_dims[2];
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/histograms/vorticity");
                dset.extend(dims);
                // spectra
                dset = data_file.openDataSet("/statistics/spectra/velocity_velocity");
                dspace = dset.getSpace();
                dspace.getSimpleExtentDims(old_dims);
                cparms = dset.getCreatePlist();
                cparms.getChunk(4, chunk_dims);
                tdim = (fs->iteration + niter_todo) - old_dims[0];
                if (tdim > 0)
                    tdim = tdim + (chunk_dims[0] - tdim % chunk_dims[0]) % chunk_dims[0];
                else
                    tdim = 0;
                dims[0] = tdim + old_dims[0];
                dims[1] = old_dims[1];
                dims[2] = old_dims[2];
                dims[3] = old_dims[3];
                dset.extend(dims);
                dset = data_file.openDataSet("/statistics/spectra/vorticity_vorticity");
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
                    double *velocity_moments = fftw_alloc_real(10*4);
                    double *vorticity_moments = fftw_alloc_real(10*4);
                    ptrdiff_t *hist_velocity = new ptrdiff_t[histogram_bins*4];
                    ptrdiff_t *hist_vorticity = new ptrdiff_t[histogram_bins*4];
                    fs->compute_velocity(fs->cvorticity);
                    fs->ift_velocity();
                    fs->compute_rspace_stats(fs->rvelocity,
                                             velocity_moments,
                                             hist_velocity,
                                             max_velocity_estimate,
                                             histogram_bins);
                    fs->ift_vorticity();
                    fs->compute_rspace_stats(fs->rvorticity,
                                             vorticity_moments,
                                             hist_vorticity,
                                             max_vorticity_estimate,
                                             histogram_bins);
                    double *spec_velocity = fftw_alloc_real(fs->nshells*9);
                    double *spec_vorticity = fftw_alloc_real(fs->nshells*9);
                    fs->cospectrum(fs->cvelocity, fs->cvelocity, spec_velocity);
                    fs->cospectrum(fs->cvorticity, fs->cvorticity, spec_vorticity);
                    if (myrank == 0)
                    {
                        H5::DataSet dset;
                        H5::DataSpace memspace, writespace;
                        hsize_t count[4], offset[4];
                        //moments
                        dset = data_file.openDataSet("statistics/moments/velocity");
                        writespace = dset.getSpace();
                        count[0] = 1;
                        count[1] = 10;
                        count[2] = 4;
                        memspace = H5::DataSpace(3, count);
                        offset[0] = fs->iteration;
                        offset[1] = 0;
                        offset[2] = 0;
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(velocity_moments, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        dset = data_file.openDataSet("statistics/moments/vorticity");
                        dset.write(vorticity_moments, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        //histograms
                        dset = data_file.openDataSet("statistics/histograms/velocity");
                        writespace = dset.getSpace();
                        count[0] = 1;
                        count[1] = histogram_bins;
                        count[2] = 4;
                        memspace = H5::DataSpace(3, count);
                        offset[0] = fs->iteration;
                        offset[1] = 0;
                        offset[2] = 0;
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(hist_velocity, H5::PredType::NATIVE_INT64, memspace, writespace);
                        dset = data_file.openDataSet("statistics/histograms/vorticity");
                        dset.write(hist_vorticity, H5::PredType::NATIVE_INT64, memspace, writespace);
                        //spectra
                        dset = data_file.openDataSet("statistics/spectra/velocity_velocity");
                        writespace = dset.getSpace();
                        count[1] = fs->nshells;
                        count[2] = 3;
                        count[3] = 3;
                        memspace = H5::DataSpace(4, count);
                        offset[3] = 0;
                        writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        dset.write(spec_velocity, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                        dset = data_file.openDataSet("statistics/spectra/vorticity_vorticity");
                        dset.write(spec_vorticity, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                    }
                    fftw_free(spec_velocity);
                    fftw_free(spec_vorticity);
                    fftw_free(velocity_moments);
                    fftw_free(vorticity_moments);
                    delete[] hist_velocity;
                    delete[] hist_vorticity;
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
                        dkx, dky, dkz,
                        dealias_type);
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
        self.particle_variables += 'tracers<{0}> *ps{1};\n'.format(self.C_dtype, self.particle_species)
        grow_template = """
                        //begincpp
                        temp_string = (std::string("/particles/") +
                                       std::string(ps{0}->name) +
                                       std::string("/{1}"));
                        dset = data_file.openDataSet(temp_string);
                        dspace = dset.getSpace();
                        dspace.getSimpleExtentDims(dims);
                        cparms = dset.getCreatePlist();
                        cparms.getChunk({2}, chunk_dims);
                        tdim = (ps{0}->iteration + {3}) - dims[0];
                        if (tdim > 0)
                            tdim = tdim + (chunk_dims[0] - tdim % chunk_dims[0]) % chunk_dims[0];
                        else
                            tdim = 0;
                        dims[0] = tdim + dims[0];
                        dset.extend(dims);
                        //endcpp
                        """
        self.file_datasets_grow += grow_template.format(self.particle_species, 'state', '3', '(niter_todo/niter_part)')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'rhs', '4', '1')
        #self.particle_definitions
        update_field = ('fs->compute_velocity(fs->cvorticity);\n' +
                        'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut) +
                        'fs->ift_velocity();\n' +
                        'ps{0}->update_field();\n').format(self.particle_species)
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        compute_acc = ('{0} *acc_field = {1}_alloc_real(ps{2}->buffered_field_descriptor->local_size);\n' +
                       '{0} *acc_field_tmp = {1}_alloc_real(fs->cd->local_size*2);\n' +
                       'fs->compute_acceleration(acc_field_tmp);\n' +
                       'ps{2}->rFFTW_to_buffered(acc_field_tmp, acc_field);\n' +
                       'ps{2}->sample_vec_field(acc_field+ps{2}->buffer_size, acceleration);\n' +
                       '{1}_free(acc_field_tmp);\n' +
                       '{1}_free(acc_field);\n').format(self.C_dtype, FFTW, self.particle_species)
        output_vel_acc =  """
                          //begincpp
                          {{
                              double *acceleration = fftw_alloc_real(ps{0}->array_size);
                              {1}
                              if (ps{0}->fs->rd->myrank == 0)
                              {{
                                  //VELOCITY begin
                                  std::string temp_string = (std::string("/particles/") +
                                                             std::string(ps{0}->name) +
                                                             std::string("/velocity"));
                                  H5::DataSet dset = data_file.openDataSet(temp_string);
                                  H5::DataSpace memspace, writespace;
                                  hsize_t count[3], offset[3];
                                  writespace = dset.getSpace();
                                  writespace.getSimpleExtentDims(count);
                                  count[0] = 1;
                                  offset[0] = ps{0}->iteration / ps{0}->traj_skip;
                                  offset[1] = 0;
                                  offset[2] = 0;
                                  memspace = H5::DataSpace(3, count);
                                  writespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                                  dset.write(ps{0}->rhs[0], H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                                  //VELOCITY end
                                  //ACCELERATION begin
                                  temp_string = (std::string("/particles/") +
                                                 std::string(ps{0}->name) +
                                                 std::string("/acceleration"));
                                  dset = data_file.openDataSet(temp_string);
                                  dset.write(acceleration, H5::PredType::NATIVE_DOUBLE, memspace, writespace);
                                  //ACCELERATION end
                              }}
                              fftw_free(acceleration);
                          }}
                          //endcpp
                          """.format(self.particle_species, compute_acc)
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
        self.particle_loop += ((update_field +
                               'ps{0}->step();\n' +
                                'if (ps{0}->iteration % niter_part == 0)\n' +
                                'ps{0}->write(&data_file, false);\n').format(self.particle_species) +
                               output_vel_acc)
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
    def get_data_file(self):
        return h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r')
    def compute_statistics(self, iter0 = 0):
        if len(list(self.statistics.keys())) > 0:
            return None
        self.read_parameters()
        with self.get_data_file() as data_file:
            if 'moments' not in data_file['statistics'].keys():
                return None
            iter0 = min(data_file['statistics/moments/velocity'].shape[0]-1, iter0)
            iter1 = data_file['iteration'].value
            self.statistics['t'] = self.parameters['dt']*np.arange(iter0, iter1+1).astype(np.float)
            self.statistics['kshell'] = data_file['kspace/kshell'].value
            self.statistics['kM'] = data_file['kspace/kM'].value
            self.statistics['energy(t, k)'] = (data_file['statistics/spectra/velocity_velocity'][iter0:iter1+1, :, 0, 0] +
                                               data_file['statistics/spectra/velocity_velocity'][iter0:iter1+1, :, 1, 1] +
                                               data_file['statistics/spectra/velocity_velocity'][iter0:iter1+1, :, 2, 2])/2
            self.statistics['enstrophy(t, k)'] = (data_file['statistics/spectra/vorticity_vorticity'][iter0:iter1+1, :, 0, 0] +
                                                  data_file['statistics/spectra/vorticity_vorticity'][iter0:iter1+1, :, 1, 1] +
                                                  data_file['statistics/spectra/vorticity_vorticity'][iter0:iter1+1, :, 2, 2])/2
            self.statistics['vel_max(t)'] = data_file['statistics/moments/velocity'][iter0:iter1+1, 9, 3]
            self.statistics['renegergy(t)'] = data_file['statistics/moments/velocity'][iter0:iter1+1, 2, 3]/2
            if self.particle_species > 0:
                self.trajectories = [
                        data_file['particles/' + key + '/state'][
                            iter0//self.parameters['niter_part']:iter1//self.parameters['niter_part']+1]
                                     for key in data_file['particles'].keys()]
            self.compute_time_averages()
        return None
    def compute_time_averages(self):
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
        self.statistics['Tint'] = 2*self.statistics['energy'] / self.statistics['diss']
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

