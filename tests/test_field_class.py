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
        self.fluid_variables += ('field<' + self.C_dtype + ', FFTW, ONE> *f;\n')
        self.fluid_start += """
                //begincpp
                f = new field<{0}, FFTW, ONE>(
                        nx, ny, nz, MPI_COMM_WORLD);
                // read rdata
                f->io("field.h5", "rdata", 0, true);
                // go to fourier space, write into cdata_tmp
                f->dft();
                f->io("field.h5", "cdata_tmp", 0, false);
                f->ift();
                f->io("field.h5", "rdata", 0, false);
                f->io("field.h5", "cdata", 0, true);
                f->ift();
                f->io("field.h5", "rdata_tmp", 0, false);
                std::vector<double> me;
                me.resize(1);
                me[0] = 30;
                hid_t gg;
                if (f->myrank == 0)
                    gg = H5Fopen("field.h5", H5F_ACC_RDWR, H5P_DEFAULT);
                f->compute_rspace_stats(
                        gg, "scal",
                        0, me);
                if (f->myrank == 0)
                    H5Fclose(gg);
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
    n = 128
    kdata = pyfftw.n_byte_align_empty(
            (n, n, n//2 + 1),
            pyfftw.simd_alignment,
            dtype = np.complex64)
    rdata = pyfftw.n_byte_align_empty(
            (n, n, n),
            pyfftw.simd_alignment,
            dtype = np.float32)
    c2r = pyfftw.FFTW(
            kdata.transpose((1, 0, 2)),
            rdata,
            axes = (0, 1, 2),
            direction = 'FFTW_BACKWARD',
            threads = 2)
    kdata[:] = bfps.tools.generate_data_3D(n, n, n, dtype = np.complex64)
    cdata = kdata.copy()
    c2r.execute()

    f = h5py.File('field.h5', 'w')
    f['cdata'] = cdata.reshape((1,) + cdata.shape)
    f['cdata_tmp'] = np.zeros(shape=(1,) + cdata.shape).astype(cdata.dtype)
    f['rdata'] = rdata.reshape((1,) + rdata.shape)
    f['rdata_tmp'] = np.zeros(shape=(1,) + rdata.shape).astype(rdata.dtype)
    f['moments/scal'] = np.zeros(shape = (1, 10)).astype(np.float)
    f['histograms/scal'] = np.zeros(shape = (1, 64)).astype(np.float)
    f.close()

    ## run cpp code
    tf = TestField()
    tf.launch(
            ['-n', '{0}'.format(n),
             '--ncpu', '2'])

    f = h5py.File('field.h5', 'r')
    err0 = np.max(np.abs(f['rdata_tmp'][0] - rdata)) / np.mean(np.abs(rdata))
    err1 = np.max(np.abs(f['rdata'][0]/(n**3) - rdata)) / np.mean(np.abs(rdata))
    err2 = np.max(np.abs(f['cdata_tmp'][0]/(n**3) - cdata)) / np.mean(np.abs(cdata))
    print(err0, err1, err2)
    assert(err0 < 1e-5)
    assert(err1 < 1e-5)
    assert(err2 < 1e-4)
    ### compare
    #fig = plt.figure(figsize=(12, 6))
    #a = fig.add_subplot(121)
    #a.set_axis_off()
    #a.imshow(rdata[0, :, :], interpolation = 'none')
    #a = fig.add_subplot(122)
    #a.set_axis_off()
    #a.imshow(f['rdata_tmp'][0, 0, :, :], interpolation = 'none')
    #fig.tight_layout()
    #fig.savefig('tst.pdf')
    # look at moments and histogram
    print('moments are ', f['moments/scal'][0])
    fig = plt.figure(figsize=(6,6))
    a = fig.add_subplot(111)
    a.plot(f['histograms/scal'][0])
    a.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tst.pdf')
    return None

if __name__ == '__main__':
    main()

