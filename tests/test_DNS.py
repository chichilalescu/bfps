from bfps.DNS import DNS


def main():
    niterations = 16
    nparticles = 100
    c = DNS(dns_type = 'NSVEp')
    c.launch(
            ['-n', '32',
             '--simname', 'vorticity_equation',
             '--np', '4',
             '--ntpp', '1',
             '--niter_todo', '{0}'.format(niterations),
             '--niter_out', '{0}'.format(niterations),
             '--niter_stat', '1',
             '--checkpoints_per_file', '{0}'.format(3),
             '--nparticles', '{0}'.format(nparticles),
             '--particle-rand-seed', '2',
             #'--njobs', '2',
             '--wd', './'])
    return None

if __name__ == '__main__':
    main()

