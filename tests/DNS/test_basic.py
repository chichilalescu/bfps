from bfps.DNS import DNS
import numpy as np
import h5py


def main():
    niterations = 32
    nparticles = 10000
    njobs = 2
    c0 = DNS()
    c0.launch(
            ['NSVEp',
             '-n', '32',
             '--simname', 'dns_nsvep',
             '--np', '4',
             '--ntpp', '1',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--checkpoints_per_file', '{0}'.format(3),
             '--nparticles', '{0}'.format(nparticles),
             '--particle-rand-seed', '2',
             '--njobs', '{0}'.format(njobs),
             '--wd', './'])
    c1 = DNS()
    c1.launch(
            ['NSVEparticles',
             '-n', '32',
             '--simname', 'dns_nsveparticles',
             '--src-simname', 'dns_nsvep',
             '--np', '4',
             '--ntpp', '1',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--checkpoints_per_file', '{0}'.format(3),
             '--nparticles', '{0}'.format(nparticles),
             '--particle-rand-seed', '2',
             '--njobs', '{0}'.format(njobs),
             '--wd', './'])
    f0 = h5py.File('dns_nsvep_checkpoint_0.h5', 'r')
    f1 = h5py.File('dns_nsveparticles_checkpoint_0.h5', 'r')
    for ii in range(0, njobs*niterations+1, niterations):
        p0 = f0['/tracers0/state/{0}'.format(ii)][:]
        p1 = f1['/tracers0/state/{0}'.format(ii)][:]
        print(np.max(np.abs(p0 - p1) / np.maximum(np.abs(p0), np.abs(p1))))
        print(np.max(np.maximum(np.abs(p0), np.abs(p1))))
    return None

if __name__ == '__main__':
    main()

