#! /usr/bin/env python

import os
import numpy as np
import h5py
import sys

import bfps
from bfps import DNS


def main():
    niterations = 32
    nparticles = 10000
    njobs = 2
    c = DNS()
    c.launch(
            ['NSVEparticles',
             '-n', '32',
             '--src-simname', 'B32p1e4',
             '--src-wd', bfps.lib_dir + '/test',
             '--src-iteration', '0',
             '--simname', 'dns_nsveparticles',
             '--np', '4',
             '--ntpp', '1',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--checkpoints_per_file', '{0}'.format(3),
             '--nparticles', '{0}'.format(nparticles),
             '--particle-rand-seed', '2',
             '--njobs', '{0}'.format(njobs),
             '--wd', './'] +
             sys.argv[1:])
    f0 = h5py.File(
            os.path.join(
                os.path.join(bfps.lib_dir, 'test'),
                'B32p1e4_checkpoint_0.h5'),
            'r')
    f1 = h5py.File(c.get_checkpoint_0_fname(), 'r')
    for iteration in [0, 32, 64]:
        field0 = f0['vorticity/complex/{0}'.format(iteration)].value
        field1 = f1['vorticity/complex/{0}'.format(iteration)].value
        assert(np.max(np.abs(field0 - field1)) < 1e-5)
        x0 = f0['tracers0/state/{0}'.format(iteration)].value
        x1 = f1['tracers0/state/{0}'.format(iteration)].value
        assert(np.max(np.abs(x0 - x1)) < 1e-5)
        y0 = f0['tracers0/rhs/{0}'.format(iteration)].value
        y1 = f1['tracers0/rhs/{0}'.format(iteration)].value
        assert(np.max(np.abs(y0 - y1)) < 1e-5)
    print('SUCCESS! Basic test passed.')
    return None

if __name__ == '__main__':
    main()

