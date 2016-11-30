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

class NSVorticityEquation(_fluid_particle_base):
    def __init__(
            self,
            name = 'NSVE-v' + bfps.__version__,
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            fftw_plan_rigor = 'FFTW_MEASURE',
            use_fftw_wisdom = True):
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
        self.fluid_output = """
                {
                    char file_name[512];
                    fs->fill_up_filename("cvorticity", file_name);
                    fs->cvorticity->io(
                        file_name,
                        "vorticity",
                        fs->iteration,
                        false);
                }
                """
        # vorticity_equation specific things
        self.includes += '#include "vorticity_equation.hpp"\n'
        self.store_kspace = """
                //begincpp
                if (myrank == 0 && iteration == 0)
                {
                    DEBUG_MSG("about to store kspace\\n");
                    TIMEZONE("fluid_base::store_kspace");
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
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->kM);
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/dk", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kk->dk);
                    H5Dclose(dset);
                    //H5Fclose(parameter_file);
                    DEBUG_MSG("finished storing kspace\\n");
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
                DEBUG_MSG("inside stat src\\n");
                hid_t stat_group;
                if (myrank == 0)
                    stat_group = H5Gopen(stat_file, "statistics", H5P_DEFAULT);
                fs->compute_velocity(fs->cvorticity);
                *tmp_vec_field = fs->cvelocity->get_cdata();
                tmp_vec_field->compute_stats(
                    fs->kk,
                    stat_group,
                    "velocity",
                    fs->iteration / niter_stat,
                    max_velocity_estimate/sqrt(3));
                DEBUG_MSG("after stats for %d\\n", fs->iteration);
                //endcpp
                """
        self.stat_src += """
                //begincpp
                *tmp_vec_field = fs->cvorticity->get_cdata();
                tmp_vec_field->compute_stats(
                    fs->kk,
                    stat_group,
                    "vorticity",
                    fs->iteration / niter_stat,
                    max_vorticity_estimate/sqrt(3));
                DEBUG_MSG("after vort stats for %d\\n", fs->iteration);
                //endcpp
                """
        self.stat_src += """
                //begincpp
                if (myrank == 0)
                    H5Gclose(stat_group);
                if (myrank == 0)
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
        self.stat_src += '}\n'
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
                DEBUG_MSG("about to allocate vorticity equation\\n");
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
                DEBUG_MSG("finished allocating stuff\\n");
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iteration;
                DEBUG_MSG("before reading\\n");
                {{
                    char file_name[512];
                    fs->fill_up_filename("cvorticity", file_name);
                    fs->cvorticity->real_space_representation = false;
                    fs->cvorticity->io(
                        file_name,
                        "vorticity",
                        fs->iteration,
                        true);
                    fs->kk->template low_pass<{0}, THREE>(fs->cvorticity->get_rdata(), fs->kk->kM);
                }}
                DEBUG_MSG("after reading\\n");
                //endcpp
                """.format(self.C_dtype, self.fftw_plan_rigor, field_H5T)
        self.fluid_start += self.store_kspace
        self.fluid_loop = ('fs->step(dt);\n' +
                           'DEBUG_MSG("this is step %d\\n", fs->iteration);\n')
        self.fluid_loop += ('if (fs->iteration % niter_out == 0)\n{\n' +
                            self.fluid_output + '\n}\n')
        self.fluid_end = ('if (fs->iteration % niter_out != 0)\n{\n' +
                          self.fluid_output + '\n}\n' +
                          'delete fs;\n' +
                          'delete tmp_vec_field;\n' +
                          'delete tmp_scal_field;\n')
        return None
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
            for k in ['t',
                      'energy(t, k)',
                      'enstrophy(t, k)',
                      'vel_max(t)',
                      'renergy(t)']:
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
            iter0 = 0):
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
        return None
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
        return None
    def prepare_launch(
            self,
            args = []):
        """Set up reasonable parameters.

        With the default Lundgren forcing applied in the band [2, 4],
        we can estimate the dissipation, therefore we can estimate
        :math:`k_M \\eta_K` and constrain the viscosity.

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
        self.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
        self.parameters['dt'] = (opt.dtfactor / opt.n)
        # custom famplitude for 288 and 576
        if opt.n == 288:
            self.parameters['famplitude'] = 0.45
        elif opt.n == 576:
            self.parameters['famplitude'] = 0.47
        if ((self.parameters['niter_todo'] % self.parameters['niter_out']) != 0):
            self.parameters['niter_out'] = self.parameters['niter_todo']
        if len(opt.src_work_dir) == 0:
            opt.src_work_dir = os.path.realpath(opt.work_dir)
        self.pars_from_namespace(opt)
        return opt
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None
    def launch_jobs(
            self,
            opt = None):
        if not os.path.exists(os.path.join(self.work_dir, self.simname + '.h5')):
            self.write_par()
            init_condition_file = os.path.join(
                    self.work_dir,
                    self.simname + '_cvorticity_i{0:0>5x}.h5'.format(0))
            if not os.path.exists(init_condition_file):
                if len(opt.src_simname) > 0:
                    src_file = os.path.join(
                            os.path.realpath(opt.src_work_dir),
                            opt.src_simname + '_cvorticity_i{0:0>5x}.h5'.format(opt.src_iteration))
                    os.symlink(src_file, init_condition_file)
                else:
                    data = self.generate_vector_field(
                           write_to_file = False,
                           spectra_slope = 2.0,
                           amplitude = 0.05)
                    f = h5py.File(init_condition_file, 'w')
                    f['vorticity/complex/{0}'.format(0)] = data
                    f.close()
        self.run(
                ncpu = opt.ncpu,
                njobs = opt.njobs,
                hours = opt.minutes // 60,
                minutes = opt.minutes % 60)
        return None

if __name__ == '__main__':
    pass

