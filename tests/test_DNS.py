from bfps.DNS import DNS


def main():
    niterations = 32
    nparticles = 10000
    c = DNS()
    c.launch(
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
             '--njobs', '2',
             '--wd', './'])
    return None

if __name__ == '__main__':
    main()

