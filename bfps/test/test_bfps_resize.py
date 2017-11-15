#! /usr/bin/env python

import os
import numpy as np
import h5py
import sys

import bfps
from bfps import DNS
from bfps import PP

import matplotlib.pyplot as plt
import pyfftw


def main():
    niterations = 2
    c = DNS()
    c.launch(
            ['NSVE',
             '-n', '32',
             '--src-simname', 'B32p1e4',
             '--src-wd', bfps.lib_dir + '/test',
             '--src-iteration', '0',
             '--simname', 'dns_test',
             '--np', '4',
             '--ntpp', '1',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--wd', './'] +
             sys.argv[1:])
    rr = PP()
    rr.launch(
            ['resize',
             '--simname', 'dns_test',
             '--new_nx', '64',
             '--new_ny', '64',
             '--new_nz', '64',
             '--new_simname', 'pp_resize_test',
             '--np', '4',
             '--ntpp', '1',
             '--iter0', '0',
             '--iter1', '{0}'.format(niterations),
             '--wd', './'] +
             sys.argv[1:])
    f0 = h5py.File(c.get_checkpoint_0_fname(), 'r')
    f1 = h5py.File('pp_resize_test_fields.h5', 'r')
    d0 = f0['vorticity/complex/0'].value
    d1 = f1['vorticity/complex/0'].value
    small_kdata = pyfftw.n_byte_align_empty(
            (32, 32, 17, 3),
            pyfftw.simd_alignment,
            dtype = c.ctype)
    small_rdata = pyfftw.n_byte_align_empty(
            (32, 32, 32, 3),
            pyfftw.simd_alignment,
            dtype = c.rtype)
    small_plan = pyfftw.FFTW(
            small_kdata.transpose((1, 0, 2, 3)),
            small_rdata,
            axes = (0, 1, 2),
            direction = 'FFTW_BACKWARD',
            threads = 4)
    big_kdata = pyfftw.n_byte_align_empty(
            (64, 64, 33, 3),
            pyfftw.simd_alignment,
            dtype = c.ctype)
    big_rdata = pyfftw.n_byte_align_empty(
            (64, 64, 64, 3),
            pyfftw.simd_alignment,
            dtype = c.rtype)
    big_plan = pyfftw.FFTW(
            big_kdata.transpose((1, 0, 2, 3)),
            big_rdata,
            axes = (0, 1, 2),
            direction = 'FFTW_BACKWARD',
            threads = 4)
    small_kdata[:] = d0
    big_kdata[:] = d1
    small_plan.execute()
    big_plan.execute()

    se = np.mean(small_rdata**2, axis = 3)**.5
    be = np.mean(big_rdata**2, axis = 3)**.5

    f = plt.figure(figsize = (6, 4))
    a = f.add_subplot(231)
    a.set_axis_off()
    a.imshow(se[0])
    a = f.add_subplot(234)
    a.set_axis_off()
    a.imshow(be[0])
    a = f.add_subplot(232)
    a.set_axis_off()
    a.imshow(se[:, 0])
    a = f.add_subplot(235)
    a.set_axis_off()
    a.imshow(be[:, 0])
    a = f.add_subplot(233)
    a.set_axis_off()
    a.imshow(se[:, :, 0])
    a = f.add_subplot(236)
    a.set_axis_off()
    a.imshow(be[:, :, 0])
    f.tight_layout()
    f.savefig('resize_test.pdf')
    plt.close(f)
    return None

if __name__ == '__main__':
    main()

