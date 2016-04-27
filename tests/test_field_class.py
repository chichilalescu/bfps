import numpy as np
import h5py
import matplotlib.pyplot as plt
import pyfftw
import bfps
import bfps.tools

import os

from bfps._fluid_base import _fluid_particle_base

class TestField(_fluid_particle_base):
    def __init__(
            self,
            name = 'TestField-v' + bfps.__version__,
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            use_fftw_wisdom = False):
        _fluid_particle_base.__init__(
                self,
                name = name + '-' + fluid_precision,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.fill_up_fluid_code()
        self.finalize_code()
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.fluid_includes += '#include "field.hpp"\n'
        self.fluid_variables += ('field<' + self.C_dtype + ', BOTH, FFTW, ONE> *f;\n')
        self.fluid_start += """
                //begincpp
                f = new field<{0}, BOTH, FFTW, ONE>(
                        nx, ny, nz, MPI_COMM_WORLD);
                DEBUG_MSG("about to read rdata\\n");
                f->io("field.h5", "rdata", 0, true);
                DEBUG_MSG("about to write rdata_tmp\\n");
                f->io("field.h5", "rdata_tmp", 0, false);
                DEBUG_MSG("about to read cdata\\n");
                f->io("field.h5", "cdata", 0, true);
                DEBUG_MSG("about to write cdata_tmp\\n");
                f->io("field.h5", "cdata_tmp", 0, false);
        //endcpp
                """.format(self.C_dtype)
        self.fluid_end += """
                //begincpp
                delete f;
                //endcpp
                """
        return None
    def specific_parser_arguments(
            self,
            parser):
        _fluid_particle_base.specific_parser_arguments(self, parser)
        return None
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args)
        self.parameters['niter_todo'] = 0
        self.pars_from_namespace(opt)
        self.set_host_info(bfps.host_info)
        self.write_par()
        self.run(ncpu = opt.ncpu)
        return None

def main():
    kdata = pyfftw.n_byte_align_empty(
            (32, 32, 17),
            pyfftw.simd_alignment,
            dtype = np.complex64)
    rdata = pyfftw.n_byte_align_empty(
            (32, 32, 32),
            pyfftw.simd_alignment,
            dtype = np.float32)
    c2r = pyfftw.FFTW(
            kdata.transpose((1, 0, 2)),
            rdata,
            axes = (0, 1, 2),
            direction = 'FFTW_BACKWARD',
            threads = 8)
    kdata[:] = bfps.tools.generate_data_3D(32, 32, 32, dtype = np.complex64)
    c2r.execute()

    f = h5py.File('field.h5', 'w')
    f['cdata'] = kdata.reshape((1,) + kdata.shape)
    f['rdata'] = rdata.reshape((1,) + rdata.shape)
    f['rdata_tmp'] = np.zeros(shape=(1,) + rdata.shape).astype(rdata.dtype)
    f['cdata_tmp'] = np.zeros(shape=(1,) + kdata.shape).astype(kdata.dtype)
    f.close()

    ## run cpp code
    tf = TestField()
    tf.launch(
            ['-n', '32',
             '--ncpu', '2'])

    f = h5py.File('field.h5', 'r')
    print(np.max(np.abs(f['rdata'].value - f['rdata_tmp'].value)))
    print(np.max(np.abs(f['cdata'].value - f['cdata_tmp'].value)))
    ## compare
    fig = plt.figure(figsize=(12, 6))
    a = fig.add_subplot(121)
    a.set_axis_off()
    a.imshow(rdata[:, 4, :], interpolation = 'none')
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(f['rdata_tmp'][0, :, 4, :], interpolation = 'none')
    fig.tight_layout()
    fig.savefig('tst.pdf')
    return None

if __name__ == '__main__':
    main()

