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



import sys
import os
import numpy as np
import h5py
import argparse

import bfps
import bfps.tools
from bfps._code import _code
from bfps._fluid_base import _fluid_particle_base

class NSVE(_fluid_particle_base):
    def __init__(
            self,
            name = 'NSVE-v' + bfps.__version__,
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            fftw_plan_rigor = 'FFTW_MEASURE',
            frozen_fields = False,
            use_fftw_wisdom = True,
            QR_stats_on = False,
            Lag_acc_stats_on = False):
        self.QR_stats_on = QR_stats_on
        self.Lag_acc_stats_on = Lag_acc_stats_on
        self.frozen_fields = frozen_fields
        self.fftw_plan_rigor = fftw_plan_rigor
        _fluid_particle_base.__init__(
                self,
                name = name + '-' + fluid_precision,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.parameters['nu'] = 0.1
        self.parameters['fmode'] = 1
        self.parameters['famplitude'] = 0.5
        self.parameters['fk0'] = 2.0
        self.parameters['fk1'] = 4.0
        self.parameters['forcing_type'] = 'linear'
        self.parameters['histogram_bins'] = 256
        self.parameters['max_velocity_estimate'] = 1.0
        self.parameters['max_vorticity_estimate'] = 1.0
        self.parameters['max_Lag_acc_estimate'] = 1.0
        self.parameters['max_pressure_estimate'] = 1.0
        self.parameters['QR2D_histogram_bins'] = 64
        self.parameters['max_trS2_estimate'] = 1.0
        self.parameters['max_Q_estimate'] = 1.0
        self.parameters['max_R_estimate'] = 1.0
        self.file_datasets_grow = """
                //begincpp
                hid_t group;
                group = H5Gopen(stat_file, "/statistics", H5P_DEFAULT);
                H5Ovisit(group, H5_INDEX_NAME, H5_ITER_NATIVE, grow_statistics_dataset, NULL);
                H5Gclose(group);
                //endcpp
                """
        self.style = {}
        self.statistics = {}
        self.fluid_output = 'fs->write(\'v\', \'c\');\n'
        # vorticity_equation specific things
        self.includes += '#include "vorticity_equation.hpp"\n'
        self.store_kspace = """
                //begincpp
                if (myrank == 0 && iteration == 0)
                {
                    TIMEZONE("fuild_base::store_kspace");
                    hsize_t dims[4];
                    hid_t space, dset;
                    // store kspace information
                    hid_t parameter_file = stat_file;
                    //char fname[256];
                    //sprintf(fname, "%s.h5", simname);
                    //parameter_file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
                    dset = H5Dopen(parameter_file, "/kspace/kshell", H5P_DEFAULT);
                    space = H5Dget_space(dset);
                    H5Sget_simple_extent_dims(space, dims, NULL);
                    H5Sclose(space);
                    if (fs->kk->nshells != dims[0])
                    {
                        DEBUG_MSG(
                            "ERROR: computed nshells %d not equal to data file nshells %d\\n",
                            fs->kk->nshells, dims[0]);
                    }
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->kshell.front());
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/nshell", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->nshell.front());
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/kM", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->kMspec);
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/dk", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->dk);
                    H5Dclose(dset);
                    //H5Fclose(parameter_file);
                }
                //endcpp
                """
        return None
    def create_stat_output(
            self,
            dset_name,
            data_buffer,
            data_type = 'H5T_NATIVE_DOUBLE',
            size_setup = None,
            close_spaces = True):
        new_stat_output_txt = 'Cdset = H5Dopen(stat_file, "{0}", H5P_DEFAULT);\n'.format(dset_name)
        if not type(size_setup) == type(None):
            new_stat_output_txt += (
                    size_setup +
                    'wspace = H5Dget_space(Cdset);\n' +
                    'ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);\n' +
                    'mspace = H5Screate_simple(ndims, count, NULL);\n' +
                    'H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);\n')
        new_stat_output_txt += ('H5Dwrite(Cdset, {0}, mspace, wspace, H5P_DEFAULT, {1});\n' +
                                'H5Dclose(Cdset);\n').format(data_type, data_buffer)
        if close_spaces:
            new_stat_output_txt += ('H5Sclose(mspace);\n' +
                                    'H5Sclose(wspace);\n')
        return new_stat_output_txt
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.stat_src += """
                //begincpp
                hid_t stat_group;
                if (myrank == 0)
                    stat_group = H5Gopen(stat_file, "statistics", H5P_DEFAULT);
                fs->compute_velocity(fs->cvorticity);
                std::vector<double> max_estimate_vector;
                max_estimate_vector.resize(4);
                *tmp_vec_field = fs->cvelocity->get_cdata();
                tmp_vec_field->compute_stats(
                    fs->kk,
                    stat_group,
                    "velocity",
                    fs->iteration / niter_stat,
                    max_velocity_estimate/sqrt(3));
                //endcpp
                """
        #if self.Lag_acc_stats_on:
        #    self.stat_src += """
        #            //begincpp
        #            tmp_vec_field->real_space_representation = false;
        #            fs->compute_Lagrangian_acceleration(tmp_vec_field->get_cdata());
        #            switch(fs->dealias_type)
        #            {
        #                case 0:
        #                    tmp_vec_field->compute_stats(
        #                        kk_two_thirds,
        #                        stat_group,
        #                        "Lagrangian_acceleration",
        #                        fs->iteration / niter_stat,
        #                        max_Lag_acc_estimate);
        #                    break;
        #                case 1:
        #                    tmp_vec_field->compute_stats(
        #                        fs->kk,
        #                        stat_group,
        #                        "Lagrangian_acceleration",
        #                        fs->iteration / niter_stat,
        #                        max_Lag_acc_estimate);
        #                    break;
        #            }
        #            tmp_scal_field->real_space_representation = false;
        #            fs->compute_velocity(fs->cvorticity);
        #            fs->ift_velocity();
        #            fs->compute_pressure(tmp_scal_field->get_cdata());
        #            switch(fs->dealias_type)
        #            {
        #                case 0:
        #                    tmp_scal_field->compute_stats(
        #                        kk_two_thirds,
        #                        stat_group,
        #                        "pressure",
        #                        fs->iteration / niter_stat,
        #                        max_pressure_estimate);
        #                    break;
        #                case 1:
        #                    tmp_scal_field->compute_stats(
        #                        fs->kk,
        #                        stat_group,
        #                        "pressure",
        #                        fs->iteration / niter_stat,
        #                        max_pressure_estimate);
        #                    break;
        #            }
        #            //endcpp
        #            """
        self.stat_src += """
                //begincpp
                *tmp_vec_field = fs->cvorticity->get_cdata();
                tmp_vec_field->compute_stats(
                    fs->kk,
                    stat_group,
                    "vorticity",
                    fs->iteration / niter_stat,
                    max_vorticity_estimate/sqrt(3));
                //endcpp
                """
        #if self.QR_stats_on:
        #    self.stat_src += """
        #        //begincpp
        #        double *trS2_Q_R_moments  = new double[10*3];
        #        double *gradu_moments     = new double[10*9];
        #        ptrdiff_t *hist_trS2_Q_R  = new ptrdiff_t[histogram_bins*3];
        #        ptrdiff_t *hist_gradu     = new ptrdiff_t[histogram_bins*9];
        #        ptrdiff_t *hist_QR2D      = new ptrdiff_t[QR2D_histogram_bins*QR2D_histogram_bins];
        #        double trS2QR_max_estimates[3];
        #        double gradu_max_estimates[9];
        #        trS2QR_max_estimates[0] = max_trS2_estimate;
        #        trS2QR_max_estimates[1] = max_Q_estimate;
        #        trS2QR_max_estimates[2] = max_R_estimate;
        #        std::fill_n(gradu_max_estimates, 9, sqrt(3*max_trS2_estimate));
        #        fs->compute_gradient_statistics(
        #            fs->cvelocity,
        #            gradu_moments,
        #            trS2_Q_R_moments,
        #            hist_gradu,
        #            hist_trS2_Q_R,
        #            hist_QR2D,
        #            trS2QR_max_estimates,
        #            gradu_max_estimates,
        #            histogram_bins,
        #            QR2D_histogram_bins);
        #        //endcpp
        #        """
        self.stat_src += """
                //begincpp
                if (myrank == 0)
                    H5Gclose(stat_group);
                if (fs->cd->myrank == 0)
                {{
                    hid_t Cdset, wspace, mspace;
                    int ndims;
                    hsize_t count[4], offset[4], dims[4];
                    offset[0] = fs->iteration/niter_stat;
                    offset[1] = 0;
                    offset[2] = 0;
                    offset[3] = 0;
                //endcpp
                """.format(self.C_dtype)
        if self.dtype == np.float32:
            field_H5T = 'H5T_NATIVE_FLOAT'
        elif self.dtype == np.float64:
            field_H5T = 'H5T_NATIVE_DOUBLE'
        self.stat_src += self.create_stat_output(
                '/statistics/xlines/velocity',
                'fs->rvelocity->get_rdata()',
                data_type = field_H5T,
                size_setup = """
                    count[0] = 1;
                    count[1] = nx;
                    count[2] = 3;
                    """,
                close_spaces = False)
        self.stat_src += self.create_stat_output(
                '/statistics/xlines/vorticity',
                'fs->rvorticity->get_rdata()',
                data_type = field_H5T)
        #if self.QR_stats_on:
        #    self.stat_src += self.create_stat_output(
        #            '/statistics/moments/trS2_Q_R',
        #            'trS2_Q_R_moments',
        #            size_setup ="""
        #                count[0] = 1;
        #                count[1] = 10;
        #                count[2] = 3;
        #                """)
        #    self.stat_src += self.create_stat_output(
        #            '/statistics/moments/velocity_gradient',
        #            'gradu_moments',
        #            size_setup ="""
        #                count[0] = 1;
        #                count[1] = 10;
        #                count[2] = 3;
        #                count[3] = 3;
        #                """)
        #    self.stat_src += self.create_stat_output(
        #            '/statistics/histograms/trS2_Q_R',
        #            'hist_trS2_Q_R',
        #            data_type = 'H5T_NATIVE_INT64',
        #            size_setup = """
        #                count[0] = 1;
        #                count[1] = histogram_bins;
        #                count[2] = 3;
        #                """)
        #    self.stat_src += self.create_stat_output(
        #            '/statistics/histograms/velocity_gradient',
        #            'hist_gradu',
        #            data_type = 'H5T_NATIVE_INT64',
        #            size_setup = """
        #                count[0] = 1;
        #                count[1] = histogram_bins;
        #                count[2] = 3;
        #                count[3] = 3;
        #                """)
        #    self.stat_src += self.create_stat_output(
        #            '/statistics/histograms/QR2D',
        #            'hist_QR2D',
        #            data_type = 'H5T_NATIVE_INT64',
        #            size_setup = """
        #                count[0] = 1;
        #                count[1] = QR2D_histogram_bins;
        #                count[2] = QR2D_histogram_bins;
        #                """)
        self.stat_src += '}\n'
        #if self.QR_stats_on:
        #    self.stat_src += """
        #        //begincpp
        #        delete[] trS2_Q_R_moments;
        #        delete[] gradu_moments;
        #        delete[] hist_trS2_Q_R;
        #        delete[] hist_gradu;
        #        delete[] hist_QR2D;
        #        //endcpp
        #        """
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += (
                'vorticity_equation<{0}, FFTW> *fs;\n'.format(self.C_dtype) +
                'field<{0}, FFTW, THREE> *tmp_vec_field;\n'.format(self.C_dtype) +
                'field<{0}, FFTW, ONE> *tmp_scal_field;\n'.format(self.C_dtype))
        self.fluid_definitions += """
                    typedef struct {{
                        {0} re;
                        {0} im;
                    }} tmp_complex_type;
                    """.format(self.C_dtype)
        self.write_fluid_stats()
        if self.dtype == np.float32:
            field_H5T = 'H5T_NATIVE_FLOAT'
        elif self.dtype == np.float64:
            field_H5T = 'H5T_NATIVE_DOUBLE'
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs = new vorticity_equation<{0}, FFTW>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz,
                        {1});
                tmp_vec_field = new field<{0}, FFTW, THREE>(
                        nx, ny, nz,
                        MPI_COMM_WORLD,
                        {1});
                tmp_scal_field = new field<{0}, FFTW, ONE>(
                        nx, ny, nz,
                        MPI_COMM_WORLD,
                        {1});
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iteration;
                fs->read('v', 'c');
                //endcpp
                """.format(self.C_dtype, self.fftw_plan_rigor, field_H5T)
        self.fluid_start += self.store_kspace
        if not self.frozen_fields:
            self.fluid_loop = 'fs->step(dt);\n'
        else:
            self.fluid_loop = ''
        self.fluid_loop += ('if (fs->iteration % niter_out == 0)\n{\n' +
                            self.fluid_output + '\n}\n')
        self.fluid_end = ('if (fs->iteration % niter_out != 0)\n{\n' +
                          self.fluid_output + '\n}\n' +
                          'delete fs;\n' +
                          'delete tmp_vec_field;\n' +
                          'delete tmp_scal_field;\n')
        return None
    #def add_3D_rFFTW_field(
    #        self,
    #        name = 'rFFTW_acc'):
    #    self.fluid_variables += 'typename fftw_interface<{0}>::complex *{1};\n'.format(self.C_dtype, name)
    #    self.fluid_start += '{0} = fftw_interface<{1}>::alloc_real(2*fs->cd->local_size);\n'.format(name, self.C_dtype)
    #    self.fluid_end   += 'fftw_interface<{0}>::free({1});\n'.format(self.C_dtype, name)
    #    return None
    #def add_interpolator(
    #        self,
    #        interp_type = 'spline',
    #        neighbours = 1,
    #        smoothness = 1,
    #        name = 'field_interpolator',
    #        field_name = 'fs->rvelocity',
    #        class_name = 'rFFTW_interpolator'):
    #    self.fluid_includes += '#include "{0}.hpp"\n'.format(class_name)
    #    self.fluid_variables += '{0} <{1}, {2}> *{3};\n'.format(
    #            class_name, self.C_dtype, neighbours, name)
    #    self.parameters[name + '_type'] = interp_type
    #    self.parameters[name + '_neighbours'] = neighbours
    #    if interp_type == 'spline':
    #        self.parameters[name + '_smoothness'] = smoothness
    #        beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
    #    elif interp_type == 'Lagrange':
    #        beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
    #    self.fluid_start += '{0} = new {1}<{2}, {3}>(fs, {4}, {5});\n'.format(
    #            name,
    #            class_name,
    #            self.C_dtype,
    #            neighbours,
    #            beta_name,
    #            field_name)
    #    self.fluid_end += 'delete {0};\n'.format(name)
    #    return None
    #def add_particles(
    #        self,
    #        integration_steps = 2,
    #        kcut = None,
    #        interpolator = 'field_interpolator',
    #        frozen_particles = False,
    #        acc_name = None,
    #        class_name = 'particles'):
    #    """Adds code for tracking a series of particle species, each
    #    consisting of `nparticles` particles.

    #    :type integration_steps: int, list of int
    #    :type kcut: None (default), str, list of str
    #    :type interpolator: str, list of str
    #    :type frozen_particles: bool
    #    :type acc_name: str

    #    .. warning:: if not None, kcut must be a list of decreasing
    #                 wavenumbers, since filtering is done sequentially
    #                 on the same complex FFTW field.
    #    """
    #    if self.dtype == np.float32:
    #        FFTW = 'fftwf'
    #    elif self.dtype == np.float64:
    #        FFTW = 'fftw'
    #    s0 = self.particle_species
    #    if type(integration_steps) == int:
    #        integration_steps = [integration_steps]
    #    if type(kcut) == str:
    #        kcut = [kcut]
    #    if type(interpolator) == str:
    #        interpolator = [interpolator]
    #    nspecies = max(len(integration_steps), len(interpolator))
    #    if type(kcut) == list:
    #        nspecies = max(nspecies, len(kcut))
    #    if len(integration_steps) == 1:
    #        integration_steps = [integration_steps[0] for s in range(nspecies)]
    #    if len(interpolator) == 1:
    #        interpolator = [interpolator[0] for s in range(nspecies)]
    #    if type(kcut) == list:
    #        if len(kcut) == 1:
    #            kcut = [kcut[0] for s in range(nspecies)]
    #    assert(len(integration_steps) == nspecies)
    #    assert(len(interpolator) == nspecies)
    #    if type(kcut) == list:
    #        assert(len(kcut) == nspecies)
    #    for s in range(nspecies):
    #        neighbours = self.parameters[interpolator[s] + '_neighbours']
    #        if type(kcut) == list:
    #            self.parameters['tracers{0}_kcut'.format(s0 + s)] = kcut[s]
    #        self.parameters['tracers{0}_interpolator'.format(s0 + s)] = interpolator[s]
    #        self.parameters['tracers{0}_acc_on'.format(s0 + s)] = int(not type(acc_name) == type(None))
    #        self.parameters['tracers{0}_integration_steps'.format(s0 + s)] = integration_steps[s]
    #        self.file_datasets_grow += """
    #                    //begincpp
    #                    group = H5Gopen(particle_file, "/tracers{0}", H5P_DEFAULT);
    #                    grow_particle_datasets(group, "", NULL, NULL);
    #                    H5Gclose(group);
    #                    //endcpp
    #                    """.format(s0 + s)

    #    #### code that outputs statistics
    #    output_vel_acc = '{\n'
    #    # array for putting sampled velocity in
    #    # must compute velocity, just in case it was messed up by some
    #    # other particle species before the stats
    #    output_vel_acc += 'fs->compute_velocity(fs->cvorticity);\n'
    #    if not type(kcut) == list:
    #        output_vel_acc += 'fs->ift_velocity();\n'
    #    if not type(acc_name) == type(None):
    #        # array for putting sampled acceleration in
    #        # must compute acceleration
    #        output_vel_acc += 'fs->compute_Lagrangian_acceleration({0});\n'.format(acc_name)
    #    for s in range(nspecies):
    #        if type(kcut) == list:
    #            output_vel_acc += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut[s])
    #            output_vel_acc += 'fs->ift_velocity();\n'
    #        output_vel_acc += """
    #            {0}->read_rFFTW(fs->rvelocity);
    #            ps{1}->sample({0}, "velocity");
    #            """.format(interpolator[s], s0 + s)
    #        if not type(acc_name) == type(None):
    #            output_vel_acc += """
    #                {0}->read_rFFTW({1});
    #                ps{2}->sample({0}, "acceleration");
    #                """.format(interpolator[s], acc_name, s0 + s)
    #    output_vel_acc += '}\n'

    #    #### initialize, stepping and finalize code
    #    if not type(kcut) == list:
    #        update_fields = ('fs->compute_velocity(fs->cvorticity);\n' +
    #                         'fs->ift_velocity();\n')
    #        self.particle_start += update_fields
    #        self.particle_loop  += update_fields
    #    else:
    #        self.particle_loop += 'fs->compute_velocity(fs->cvorticity);\n'
    #    self.particle_includes += '#include "{0}.hpp"\n'.format(class_name)
    #    self.particle_stat_src += (
    #            'if (ps0->iteration % niter_part == 0)\n' +
    #            '{\n')
    #    for s in range(nspecies):
    #        neighbours = self.parameters[interpolator[s] + '_neighbours']
    #        self.particle_start += 'sprintf(fname, "tracers{0}");\n'.format(s0 + s)
    #        self.particle_end += ('ps{0}->write();\n' +
    #                              'delete ps{0};\n').format(s0 + s)
    #        self.particle_variables += '{0}<VELOCITY_TRACER, {1}, {2}> *ps{3};\n'.format(
    #                class_name,
    #                self.C_dtype,
    #                neighbours,
    #                s0 + s)
    #        self.particle_start += ('ps{0} = new {1}<VELOCITY_TRACER, {2}, {3}>(\n' +
    #                                'fname, particle_file, {4},\n' +
    #                                'niter_part, tracers{0}_integration_steps);\n').format(
    #                                        s0 + s,
    #                                        class_name,
    #                                        self.C_dtype,
    #                                        neighbours,
    #                                        interpolator[s])
    #        self.particle_start += ('ps{0}->dt = dt;\n' +
    #                                'ps{0}->iteration = iteration;\n' +
    #                                'ps{0}->read();\n').format(s0 + s)
    #        if not frozen_particles:
    #            if type(kcut) == list:
    #                update_field = ('fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut[s]) +
    #                                'fs->ift_velocity();\n')
    #                self.particle_loop += update_field
    #            self.particle_loop += '{0}->read_rFFTW(fs->rvelocity);\n'.format(interpolator[s])
    #            self.particle_loop += 'ps{0}->step();\n'.format(s0 + s)
    #        self.particle_stat_src += 'ps{0}->write(false);\n'.format(s0 + s)
    #    self.particle_stat_src += output_vel_acc
    #    self.particle_stat_src += '}\n'
    #    self.particle_species += nspecies
    #    return None
    def get_postprocess_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_postprocess.h5')
    def get_postprocess_file(self):
        return h5py.File(self.get_postprocess_file_name(), 'r')
    def compute_statistics(self, iter0 = 0, iter1 = None):
        """Run basic postprocessing on raw data.
        The energy spectrum :math:`E(t, k)` and the enstrophy spectrum
        :math:`\\frac{1}{2}\omega^2(t, k)` are computed from the

        .. math::

            \sum_{k \\leq \\|\\mathbf{k}\\| \\leq k+dk}\\hat{u_i} \\hat{u_j}^*, \\hskip .5cm
            \sum_{k \\leq \\|\\mathbf{k}\\| \\leq k+dk}\\hat{\omega_i} \\hat{\\omega_j}^*

        tensors, and the enstrophy spectrum is also used to
        compute the dissipation :math:`\\varepsilon(t)`.
        These basic quantities are stored in a newly created HDF5 file,
        ``simname_postprocess.h5``.
        """
        if len(list(self.statistics.keys())) > 0:
            return None
        self.read_parameters()
        with self.get_data_file() as data_file:
            if 'moments' not in data_file['statistics'].keys():
                return None
            iter0 = min((data_file['statistics/moments/velocity'].shape[0] *
                         self.parameters['niter_stat']-1),
                        iter0)
            if type(iter1) == type(None):
                iter1 = data_file['iteration'].value
            else:
                iter1 = min(data_file['iteration'].value, iter1)
            ii0 = iter0 // self.parameters['niter_stat']
            ii1 = iter1 // self.parameters['niter_stat']
            self.statistics['kshell'] = data_file['kspace/kshell'].value
            self.statistics['kM'] = data_file['kspace/kM'].value
            self.statistics['dk'] = data_file['kspace/dk'].value
            computation_needed = True
            pp_file = h5py.File(self.get_postprocess_file_name(), 'a')
            if 'ii0' in pp_file.keys():
                computation_needed =  not (ii0 == pp_file['ii0'].value and
                                           ii1 == pp_file['ii1'].value)
                if computation_needed:
                    for k in pp_file.keys():
                        del pp_file[k]
            if computation_needed:
                pp_file['iter0'] = iter0
                pp_file['iter1'] = iter1
                pp_file['ii0'] = ii0
                pp_file['ii1'] = ii1
                pp_file['t'] = (self.parameters['dt']*
                                self.parameters['niter_stat']*
                                (np.arange(ii0, ii1+1).astype(np.float)))
                pp_file['energy(t, k)'] = (
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 0, 0] +
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 1, 1] +
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 2, 2])/2
                pp_file['enstrophy(t, k)'] = (
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 0, 0] +
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 1, 1] +
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 2, 2])/2
                pp_file['vel_max(t)'] = data_file['statistics/moments/velocity']  [ii0:ii1+1, 9, 3]
                pp_file['renergy(t)'] = data_file['statistics/moments/velocity'][ii0:ii1+1, 2, 3]/2
                if 'trS2_Q_R' in data_file['statistics/moments'].keys():
                    pp_file['mean_trS2(t)'] = data_file['statistics/moments/trS2_Q_R'][:, 1, 0]
            for k in ['t',
                      'energy(t, k)',
                      'enstrophy(t, k)',
                      'vel_max(t)',
                      'renergy(t)',
                      'mean_trS2(t)']:
                if k in pp_file.keys():
                    self.statistics[k] = pp_file[k].value
            self.compute_time_averages()
        return None
    def compute_time_averages(self):
        """Compute easy stats.

        Further computation of statistics based on the contents of
        ``simname_postprocess.h5``.
        Standard quantities are as follows
        (consistent with [Ishihara]_):

        .. math::

            U_{\\textrm{int}}(t) = \\sqrt{\\frac{2E(t)}{3}}, \\hskip .5cm
            L_{\\textrm{int}}(t) = \\frac{\pi}{2U_{int}^2(t)} \\int \\frac{dk}{k} E(t, k), \\hskip .5cm
            T_{\\textrm{int}}(t) =
            \\frac{L_{\\textrm{int}}(t)}{U_{\\textrm{int}}(t)}

            \\eta_K = \\left(\\frac{\\nu^3}{\\varepsilon}\\right)^{1/4}, \\hskip .5cm
            \\tau_K = \\left(\\frac{\\nu}{\\varepsilon}\\right)^{1/2}, \\hskip .5cm
            \\lambda = \\sqrt{\\frac{15 \\nu U_{\\textrm{int}}^2}{\\varepsilon}}

            Re = \\frac{U_{\\textrm{int}} L_{\\textrm{int}}}{\\nu}, \\hskip
            .5cm
            R_{\\lambda} = \\frac{U_{\\textrm{int}} \\lambda}{\\nu}

        .. [Ishihara] T. Ishihara et al,
                      *Small-scale statistics in high-resolution direct numerical
                      simulation of turbulence: Reynolds number dependence of
                      one-point velocity gradient statistics*.
                      J. Fluid Mech.,
                      **592**, 335-366, 2007
        """
        for key in ['energy', 'enstrophy']:
            self.statistics[key + '(t)'] = (self.statistics['dk'] *
                                            np.sum(self.statistics[key + '(t, k)'], axis = 1))
        self.statistics['Uint(t)'] = np.sqrt(2*self.statistics['energy(t)'] / 3)
        self.statistics['Lint(t)'] = ((self.statistics['dk']*np.pi /
                                       (2*self.statistics['Uint(t)']**2)) *
                                      np.nansum(self.statistics['energy(t, k)'] /
                                                self.statistics['kshell'][None, :], axis = 1))
        for key in ['energy',
                    'enstrophy',
                    'vel_max',
                    'mean_trS2',
                    'Uint',
                    'Lint']:
            if key + '(t)' in self.statistics.keys():
                self.statistics[key] = np.average(self.statistics[key + '(t)'], axis = 0)
        for suffix in ['', '(t)']:
            self.statistics['diss'    + suffix] = (self.parameters['nu'] *
                                                   self.statistics['enstrophy' + suffix]*2)
            self.statistics['etaK'    + suffix] = (self.parameters['nu']**3 /
                                                   self.statistics['diss' + suffix])**.25
            self.statistics['tauK'    + suffix] =  (self.parameters['nu'] /
                                                    self.statistics['diss' + suffix])**.5
            self.statistics['Re' + suffix] = (self.statistics['Uint' + suffix] *
                                              self.statistics['Lint' + suffix] /
                                              self.parameters['nu'])
            self.statistics['lambda' + suffix] = (15 * self.parameters['nu'] *
                                                  self.statistics['Uint' + suffix]**2 /
                                                  self.statistics['diss' + suffix])**.5
            self.statistics['Rlambda' + suffix] = (self.statistics['Uint' + suffix] *
                                                   self.statistics['lambda' + suffix] /
                                                   self.parameters['nu'])
            self.statistics['kMeta' + suffix] = (self.statistics['kM'] *
                                                 self.statistics['etaK' + suffix])
            if self.parameters['dealias_type'] == 1:
                self.statistics['kMeta' + suffix] *= 0.8
        self.statistics['Tint'] = self.statistics['Lint'] / self.statistics['Uint']
        self.statistics['Taylor_microscale'] = self.statistics['lambda']
        return None
    def set_plt_style(
            self,
            style = {'dashes' : (None, None)}):
        self.style.update(style)
        return None
    def read_cfield(
            self,
            field_name = 'vorticity',
            iteration = 0):
        """read the Fourier representation of a vector field.

        Read the binary file containing iteration ``iteration`` of the
        field ``field_name``, and return it as a properly shaped
        ``numpy.memmap`` object.
        """
        return np.memmap(
                os.path.join(self.work_dir,
                             self.simname + '_{0}_i{1:0>5x}'.format('c' + field_name, iteration)),
                dtype = self.ctype,
                mode = 'r',
                shape = (self.parameters['ny'],
                         self.parameters['nz'],
                         self.parameters['nx']//2+1,
                         3))
    def write_par(
            self,
            iter0 = 0,
            particle_ic = None):
        _fluid_particle_base.write_par(self, iter0 = iter0)
        with h5py.File(self.get_data_file_name(), 'r+') as ofile:
            kspace = self.get_kspace()
            nshells = kspace['nshell'].shape[0]
            vec_stat_datasets = ['velocity', 'vorticity']
            scal_stat_datasets = []
            for k in vec_stat_datasets:
                time_chunk = 2**20//(8*3*self.parameters['nx']) # FIXME: use proper size of self.dtype
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/xlines/' + k,
                                     (1, self.parameters['nx'], 3),
                                     chunks = (time_chunk, self.parameters['nx'], 3),
                                     maxshape = (None, self.parameters['nx'], 3),
                                     dtype = self.dtype)
            if self.Lag_acc_stats_on:
                vec_stat_datasets += ['Lagrangian_acceleration']
                scal_stat_datasets += ['pressure']
            for k in vec_stat_datasets:
                time_chunk = 2**20//(8*3*3*nshells)
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/spectra/' + k + '_' + k,
                                     (1, nshells, 3, 3),
                                     chunks = (time_chunk, nshells, 3, 3),
                                     maxshape = (None, nshells, 3, 3),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*4*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/' + k,
                                     (1, 10, 4),
                                     chunks = (time_chunk, 10, 4),
                                     maxshape = (None, 10, 4),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*4*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/' + k,
                                     (1,
                                      self.parameters['histogram_bins'],
                                      4),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               4),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 4),
                                     dtype = np.int64)
            for k in scal_stat_datasets:
                time_chunk = 2**20//(8*nshells)
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/spectra/' + k + '_' + k,
                                     (1, nshells),
                                     chunks = (time_chunk, nshells),
                                     maxshape = (None, nshells),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/' + k,
                                     (1, 10),
                                     chunks = (time_chunk, 10),
                                     maxshape = (None, 10),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/' + k,
                                     (1,
                                      self.parameters['histogram_bins']),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins']),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins']),
                                     dtype = np.int64)
            if self.QR_stats_on:
                time_chunk = 2**20//(8*3*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/trS2_Q_R',
                                     (1,
                                      self.parameters['histogram_bins'],
                                      3),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               3),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 3),
                                     dtype = np.int64)
                time_chunk = 2**20//(8*9*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/velocity_gradient',
                                     (1,
                                      self.parameters['histogram_bins'],
                                      3,
                                      3),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               3,
                                               3),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 3,
                                                 3),
                                     dtype = np.int64)
                time_chunk = 2**20//(8*3*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/trS2_Q_R',
                                     (1, 10, 3),
                                     chunks = (time_chunk, 10, 3),
                                     maxshape = (None, 10, 3),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*9*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/velocity_gradient',
                                     (1, 10, 3, 3),
                                     chunks = (time_chunk, 10, 3, 3),
                                     maxshape = (None, 10, 3, 3),
                                     dtype = np.float64)
                time_chunk = 2**20//(8*self.parameters['QR2D_histogram_bins']**2)
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/QR2D',
                                     (1,
                                      self.parameters['QR2D_histogram_bins'],
                                      self.parameters['QR2D_histogram_bins']),
                                     chunks = (time_chunk,
                                               self.parameters['QR2D_histogram_bins'],
                                               self.parameters['QR2D_histogram_bins']),
                                     maxshape = (None,
                                                 self.parameters['QR2D_histogram_bins'],
                                                 self.parameters['QR2D_histogram_bins']),
                                     dtype = np.int64)
        if self.particle_species == 0:
            return None

        if type(particle_ic) == type(None):
            pbase_shape = (self.parameters['nparticles'],)
            number_of_particles = self.parameters['nparticles']
        else:
            pbase_shape = particle_ic.shape[:-1]
            assert(particle_ic.shape[-1] == 3)
            number_of_particles = 1
            for val in pbase_shape[1:]:
                number_of_particles *= val

        with h5py.File(self.get_particle_file_name(), 'a') as ofile:
            for s in range(self.particle_species):
                ofile.create_group('tracers{0}'.format(s))
                time_chunk = 2**20 // (8*3*number_of_particles)
                time_chunk = max(time_chunk, 1)
                dims = ((1,
                         self.parameters['tracers{0}_integration_steps'.format(s)]) +
                        pbase_shape + (3,))
                maxshape = (h5py.h5s.UNLIMITED,) + dims[1:]
                if len(pbase_shape) > 1:
                    chunks = (time_chunk, 1, 1) + dims[3:]
                else:
                    chunks = (time_chunk, 1) + dims[2:]
                chunks = (time_chunk, 1, 1) + dims[3:]
                bfps.tools.create_alloc_early_dataset(
                        ofile,
                        '/tracers{0}/rhs'.format(s),
                        dims, maxshape, chunks)
                if len(pbase_shape) > 1:
                    chunks = (time_chunk, 1) + pbase_shape[1:] + (3,)
                else:
                    chunks = (time_chunk, pbase_shape[0], 3)
                bfps.tools.create_alloc_early_dataset(
                        ofile,
                        '/tracers{0}/state'.format(s),
                        (1,) + pbase_shape + (3,),
                        (h5py.h5s.UNLIMITED,) + pbase_shape + (3,),
                        chunks)
                bfps.tools.create_alloc_early_dataset(
                        ofile,
                        '/tracers{0}/velocity'.format(s),
                        (1,) + pbase_shape + (3,),
                        (h5py.h5s.UNLIMITED,) + pbase_shape + (3,),
                        chunks,
                        dset_dtype = h5py.h5t.IEEE_F32LE)
                if self.parameters['tracers{0}_acc_on'.format(s)]:
                    bfps.tools.create_alloc_early_dataset(
                            ofile,
                            '/tracers{0}/acceleration'.format(s),
                            (1,) + pbase_shape + (3,),
                            (h5py.h5s.UNLIMITED,) + pbase_shape + (3,),
                            chunks,
                            dset_dtype = h5py.h5t.IEEE_F32LE)
        return None
    #def add_particle_fields(
    #        self,
    #        interp_type = 'spline',
    #        kcut = None,
    #        neighbours = 1,
    #        smoothness = 1,
    #        name = 'particle_field',
    #        field_class = 'rFFTW_interpolator',
    #        acc_field_name = 'rFFTW_acc'):
    #    self.fluid_includes += '#include "{0}.hpp"\n'.format(field_class)
    #    self.fluid_variables += field_class + '<{0}, {1}> *vel_{2}, *acc_{2};\n'.format(
    #            self.C_dtype, neighbours, name)
    #    self.parameters[name + '_type'] = interp_type
    #    self.parameters[name + '_neighbours'] = neighbours
    #    if interp_type == 'spline':
    #        self.parameters[name + '_smoothness'] = smoothness
    #        beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
    #    elif interp_type == 'Lagrange':
    #        beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
    #    if field_class == 'rFFTW_interpolator':
    #        self.fluid_start += ('vel_{0} = new {1}<{2}, {3}>(fs, {4}, fs->rvelocity);\n' +
    #                             'acc_{0} = new {1}<{2}, {3}>(fs, {4}, {5});\n').format(name,
    #                                                                               field_class,
    #                                                                               self.C_dtype,
    #                                                                               neighbours,
    #                                                                               beta_name,
    #                                                                               acc_field_name)
    #    elif field_class == 'interpolator':
    #        self.fluid_start += ('vel_{0} = new {1}<{2}, {3}>(fs, {4});\n' +
    #                             'acc_{0} = new {1}<{2}, {3}>(fs, {4});\n').format(name,
    #                                                                               field_class,
    #                                                                               self.C_dtype,
    #                                                                               neighbours,
    #                                                                               beta_name,
    #                                                                               acc_field_name)
    #    self.fluid_end += ('delete vel_{0};\n' +
    #                       'delete acc_{0};\n').format(name)
    #    update_fields = 'fs->compute_velocity(fs->cvorticity);\n'
    #    if not type(kcut) == type(None):
    #        update_fields += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut)
    #    update_fields += ('fs->ift_velocity();\n' +
    #                      'fs->compute_Lagrangian_acceleration(acc_{0}->field);\n').format(name)
    #    self.fluid_start += update_fields
    #    self.fluid_loop += update_fields
    #    return None
    def specific_parser_arguments(
            self,
            parser):
        _fluid_particle_base.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--src-wd',
                type = str,
                dest = 'src_work_dir',
                default = '')
        parser.add_argument(
                '--src-simname',
                type = str,
                dest = 'src_simname',
                default = '')
        parser.add_argument(
                '--src-iteration',
                type = int,
                dest = 'src_iteration',
                default = 0)
        parser.add_argument(
               '--njobs',
               type = int, dest = 'njobs',
               default = 1)
        parser.add_argument(
               '--QR-stats',
               action = 'store_true',
               dest = 'QR_stats',
               help = 'add this option if you want to compute velocity gradient and QR stats')
        parser.add_argument(
               '--Lag-acc-stats',
               action = 'store_true',
               dest = 'Lag_acc_stats',
               help = 'add this option if you want to compute Lagrangian acceleration statistics')
        parser.add_argument(
               '--kMeta',
               type = float,
               dest = 'kMeta',
               default = 2.0)
        parser.add_argument(
               '--dtfactor',
               type = float,
               dest = 'dtfactor',
               default = 0.5,
               help = 'dt is computed as DTFACTOR / N')
        parser.add_argument(
               '--particle-rand-seed',
               type = int,
               dest = 'particle_rand_seed',
               default = None)
        parser.add_argument(
               '--pclouds',
               type = int,
               dest = 'pclouds',
               default = 1,
               help = ('number of particle clouds. Particle "clouds" '
                       'consist of particles distributed according to '
                       'pcloud-type.'))
        parser.add_argument(
                '--pcloud-type',
                choices = ['random-cube',
                           'regular-cube'],
                dest = 'pcloud_type',
                default = 'random-cube')
        parser.add_argument(
               '--particle-cloud-size',
               type = float,
               dest = 'particle_cloud_size',
               default = 2*np.pi)
        parser.add_argument(
                '--neighbours',
                type = int,
                dest = 'neighbours',
                default = 1)
        parser.add_argument(
                '--smoothness',
                type = int,
                dest = 'smoothness',
                default = 1)
        return None
    def prepare_launch(
            self,
            args = []):
        """Set up reasonable parameters.

        With the default Lundgren forcing applied in the band [2, 4],
        we can estimate the dissipation, therefore we can estimate
        :math:`k_M \\eta_K` and constrain the viscosity.
        Also, if velocity gradient statistics are computed, the
        dissipation is used for estimating the bins of the QR histogram.

        In brief, the command line parameter :math:`k_M \\eta_K` is
        used in the following formula for :math:`\\nu` (:math:`N` is the
        number of real space grid points per coordinate):

        .. math::

            \\nu = \\left(\\frac{2 k_M \\eta_K}{N} \\right)^{4/3}

        With this choice, the average dissipation :math:`\\varepsilon`
        will be close to 0.4, and the integral scale velocity will be
        close to 0.77, yielding the approximate value for the Taylor
        microscale and corresponding Reynolds number:

        .. math::

            \\lambda \\approx 4.75\\left(\\frac{2 k_M \\eta_K}{N} \\right)^{4/6}, \\hskip .5in
            R_\\lambda \\approx 3.7 \\left(\\frac{N}{2 k_M \\eta_K} \\right)^{4/6}

        """
        opt = _code.prepare_launch(self, args = args)
        self.QR_stats_on = opt.QR_stats
        self.Lag_acc_stats_on = opt.Lag_acc_stats
        self.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
        self.parameters['dt'] = (opt.dtfactor / opt.n)
        # custom famplitude for 288 and 576
        if opt.n == 288:
            self.parameters['famplitude'] = 0.45
        elif opt.n == 576:
            self.parameters['famplitude'] = 0.47
        if ((self.parameters['niter_todo'] % self.parameters['niter_out']) != 0):
            self.parameters['niter_out'] = self.parameters['niter_todo']
        if self.QR_stats_on:
            # max_Q_estimate and max_R_estimate are just used for the 2D pdf
            # therefore I just want them to be small multiples of mean trS2
            # I'm already estimating the dissipation with kMeta...
            meantrS2 = (opt.n//2 / opt.kMeta)**4 * self.parameters['nu']**2
            self.parameters['max_Q_estimate'] = meantrS2
            self.parameters['max_R_estimate'] = .4*meantrS2**1.5
            # add QR suffix to code name, since we now expect additional
            # datasets in the .h5 file
            self.name += '-QR'
        if self.Lag_acc_stats_on:
            self.name += '-Lag_acc'
        if len(opt.src_work_dir) == 0:
            opt.src_work_dir = os.path.realpath(opt.work_dir)
        self.pars_from_namespace(opt)
        return opt
    def launch(
            self,
            args = [],
            noparticles = False,
            **kwargs):
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        if noparticles:
            opt.nparticles = 0
        elif type(opt.nparticles) == int:
            if opt.nparticles > 0:
                self.name += '-particles'
                self.add_3D_rFFTW_field(
                        name = 'rFFTW_acc')
                self.add_interpolator(
                        name = 'cubic_spline',
                        neighbours = opt.neighbours,
                        smoothness = opt.smoothness,
                        class_name = 'rFFTW_interpolator')
                self.add_particles(
                        integration_steps = [4],
                        interpolator = 'cubic_spline',
                        acc_name = 'rFFTW_acc',
                        class_name = 'rFFTW_distributed_particles')
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None
    def launch_jobs(
            self,
            opt = None):
        if not os.path.exists(os.path.join(self.work_dir, self.simname + '.h5')):
            particle_initial_condition = None
            if opt.pclouds > 1:
                np.random.seed(opt.particle_rand_seed)
                if opt.pcloud_type == 'random-cube':
                    particle_initial_condition = (
                        np.random.random((opt.pclouds, 1, 3))*2*np.pi +
                        np.random.random((1, self.parameters['nparticles'], 3))*opt.particle_cloud_size)
                elif opt.pcloud_type == 'regular-cube':
                    onedarray = np.linspace(
                            -opt.particle_cloud_size/2,
                            opt.particle_cloud_size/2,
                            self.parameters['nparticles'])
                    particle_initial_condition = np.zeros(
                            (opt.pclouds,
                             self.parameters['nparticles'],
                             self.parameters['nparticles'],
                             self.parameters['nparticles'], 3),
                            dtype = np.float64)
                    particle_initial_condition[:] = \
                        np.random.random((opt.pclouds, 1, 1, 1, 3))*2*np.pi
                    particle_initial_condition[..., 0] += onedarray[None, None, None, :]
                    particle_initial_condition[..., 1] += onedarray[None, None, :, None]
                    particle_initial_condition[..., 2] += onedarray[None, :, None, None]
            self.write_par(
                    particle_ic = particle_initial_condition)
            if self.parameters['nparticles'] > 0:
                data = self.generate_tracer_state(
                        species = 0,
                        rseed = opt.particle_rand_seed,
                        data = particle_initial_condition)
                for s in range(1, self.particle_species):
                    self.generate_tracer_state(species = s, data = data)
            init_condition_file = os.path.join(
                    self.work_dir,
                    self.simname + '_cvorticity_i{0:0>5x}'.format(0))
            if not os.path.exists(init_condition_file):
                if len(opt.src_simname) > 0:
                    src_file = os.path.join(
                            os.path.realpath(opt.src_work_dir),
                            opt.src_simname + '_cvorticity_i{0:0>5x}'.format(opt.src_iteration))
                    os.symlink(src_file, init_condition_file)
                else:
                   self.generate_vector_field(
                           write_to_file = True,
                           spectra_slope = 2.0,
                           amplitude = 0.05)
        self.run(
                ncpu = opt.ncpu,
                njobs = opt.njobs,
                hours = opt.minutes // 60,
                minutes = opt.minutes % 60)
        return None

def main():
    c = NSVE()
    c.launch(
            ['-n', '32',
             '--ncpu', '4',
             '--niter_todo', '48',
             '--wd', 'data/single'] +
            sys.argv[1:])
    return None

if __name__ == '__main__':
    main()

