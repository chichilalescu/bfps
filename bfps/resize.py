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

class vorticity_resize(bfps.code):
    def __init__(
            self,
            name = 'vorticity_resize',
            work_dir = './'):
        super(vorticity_resize, self).__init__()
        self.work_dir = work_dir
        self.particle_species = 0
        self.name = name
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['dst_iter'] = 0
        self.parameters['dst_nx'] = 32
        self.parameters['dst_ny'] = 32
        self.parameters['dst_nz'] = 32
        self.parameters['dst_simname'] = 'new_test'
        self.parameters['dst_dkx'] = 1.0
        self.parameters['dst_dky'] = 1.0
        self.parameters['dst_dkz'] = 1.0
        self.fill_up_code()
        return None
    def fill_up_code(self):
        self.includes = '#include "fluid_solver.hpp"\n'
        self.variables  += self.cdef_pars()
        self.definitions+= self.cread_pars()
        self.includes += '#include <cstring>\n'
        self.includes += '#include "fftw_tools.hpp"\n'
        self.variables += 'fluid_solver<float> *fs0, *fs1;\n'
        self.main = """
                //begincpp
                char fname[512];
                fs0 = new fluid_solver<float>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs1 = new fluid_solver<float>(
                        dst_simname,
                        dst_nx, dst_ny, dst_nz,
                        dst_dkx, dst_dky, dst_dkz);
                fs0->iteration = iter0;
                fs0->read('v', 'c');
                fs0->low_pass_Fourier(fs0->cvorticity, 3, fs0->kM);
                fs0->force_divfree(fs0->cvorticity);
                fs0->symmetrize(fs0->cvorticity, 3);
                fs0->write('v', 'r');
                fs0->write('u', 'r');
                copy_complex_array(fs0->cd, fs0->cvorticity,
                                   fs1->cd, fs1->cvorticity,
                                   3);
                fs1->write('v', 'c');
                fs1->write('v', 'r');
                fs1->write('u', 'r');
                delete fs0;
                delete fs1;
                //endcpp
                """
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
            filename = os.path.join(self.work_dir, simname + '_' + field + '_i{0:0>5x}'.format(iteration))
        Rdata0 = np.fromfile(
                filename,
                dtype = np.float32).reshape((-1,
                                             self.parameters['ny'],
                                             self.parameters['nx'], 3))
        energy = np.sum(Rdata0[:, yval, :, :]**2, axis = 2)*.5
        axis.imshow(energy, interpolation='none')
        axis.set_title('{0} {1}'.format(field,
                                        np.average(Rdata0[..., 0]**2 +
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

def double(opt):
    if opt.run or opt.clean:
        subprocess.call(['rm {0}/test_*'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*.pickle'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*.elf'.format(opt.work_dir)], shell = True)
        subprocess.call(['rm {0}/*version_info.txt'.format(opt.work_dir)], shell = True)
    if opt.clean:
        subprocess.call(['make', 'clean'])
        return None
    c = vorticity_resize(work_dir = opt.work_dir)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = 'test2'
    c.write_src()
    c.write_par(simname = 'test1')
    if opt.run:
        #c.generate_vector_field(simname = 'test1')
        c.run(ncpu = opt.ncpu,
              simname = 'test1')

    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(221)
    c.plot_vel_cut(
            a,
            simname = 'test1',
            field = 'rvorticity',
            iteration = 0,
            yval = 13)
    a = fig.add_subplot(222)
    c.plot_vel_cut(
            a,
            simname = 'test1',
            field = 'rvelocity',
            iteration = 0,
            yval = 13)
    c.parameters['nx'] *= 2
    c.parameters['ny'] *= 2
    c.parameters['nz'] *= 2
    a = fig.add_subplot(223)
    c.plot_vel_cut(
            a,
            simname = 'test2',
            field = 'rvorticity',
            iteration = 0,
            yval = 26)
    a = fig.add_subplot(224)
    c.plot_vel_cut(
            a,
            simname = 'test2',
            field = 'rvelocity',
            iteration = 0,
            yval = 26)
    fig.savefig('resize.pdf', format = 'pdf')
    return None

