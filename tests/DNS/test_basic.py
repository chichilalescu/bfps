from bfps.DNS import DNS
import numpy as np
import h5py


def main():
    niterations = 32
    nparticles = 10000
    njobs = 2
    c = DNS()
    c.launch(
            ['NSVEparticles',
             '-n', '32',
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
             '--wd', './'])
    return None

if __name__ == '__main__':
    main()

