import sys
import numpy as np
import argparse
import os

import bfps

def get_DNS_parameters(
        DNS_type = 'A',
        N = 512,
        nnodes = 1,
        nprocesses = 1,
        output_on = False,
        cores_per_node = 16,
        nparticles = int(1e5)):
    simname = (DNS_type + '{0:0>4d}'.format(N))
    if output_on:
        simname = DNS_type + simname
    class_name = 'NSVE'
    if DNS_type != 'A':
        simname += 'p{0}e{1}'.format(
                int(nparticles / 10**np.log10(nparticles)),
                int(np.log10(nparticles)))
        class_name += 'particles'
    work_dir = 'nn{0:0>4d}np{1}'.format(nnodes, nprocesses)
    if not output_on:
        class_name += '_no_output'
    src_simname = 'N{0:0>4d}_kMeta2'.format(N)
    src_iteration = -1
    if N == 512:
        src_iteration = 3072
    if N == 1024:
        src_iteration = 0x4000
    if N == 2048:
        src_iteration = 0x6000
    if N == 4096:
        src_iteration = 0
    DNS_parameters = [
            class_name,
            '-n', '{0}'.format(N),
            '--np', '{0}'.format(nnodes*nprocesses),
            '--ntpp', '{0}'.format(cores_per_node // nprocesses),
            '--simname', simname,
            '--wd', work_dir,
            '--niter_todo', '12',
            '--niter_out', '12',
            '--niter_stat', '3']
    if src_iteration >= 0:
        DNS_parameters += [
            '--src-wd', 'database',
            '--src-simname', src_simname,
            '--src-iteration', '{0}'.format(src_iteration)]
    if DNS_type != 'A':
        DNS_parameters += [
                '--nparticles', '{0}'.format(nparticles)]
        nneighbours = np.where(np.array(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']) == DNS_type)[0][0]
        if nneighbours < 3:
            smoothness = 1
        else:
            smoothness = 2
        DNS_parameters += [
                '--tracers0_neighbours', '{0}'.format(nneighbours),
                '--tracers0_smoothness', '{0}'.format(smoothness),
                '--particle-rand-seed', '2']
    return simname, work_dir, DNS_parameters

def main():
        #DNS_type = 'A',
        #N = 512,
        #nnodes = 1,
        #nprocesses = 1,
        #output_on = False,
        #cores_per_node = 16,
        #nparticles = 1e5)
    parser = argparse.ArgumentParser(prog = 'launcher')
    parser.add_argument(
            'DNS_setup',
            choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
            type = str)
    parser.add_argument(
            '-n',
            type = int,
            dest = 'n',
            default = 32)
    parser.add_argument(
            '--nnodes',
            type = int,
            dest = 'nnodes',
            default = 1)
    parser.add_argument(
            '--nprocesses',
            type = int,
            dest = 'nprocesses',
            default = 1)
    parser.add_argument(
            '--ncores',
            type = int,
            dest = 'ncores',
            default = 4)
    parser.add_argument(
            '--output-on',
            action = 'store_true',
            dest = 'output_on')
    parser.add_argument(
            '--nparticles',
            type = int,
            dest = 'nparticles',
            default = int(1e5))
    opt = parser.parse_args(sys.argv[1:])
    simname, work_dir, params = get_DNS_parameters(
            DNS_type = opt.DNS_setup,
            N = opt.n,
            nnodes = opt.nnodes,
            nprocesses = opt.nprocesses,
            output_on = opt.output_on,
            nparticles = opt.nparticles,
            cores_per_node = opt.ncores)
    print(work_dir + '/' + simname)
    print(' '.join(params))
    # these following 2 lines actually launch something
    # I'm not passing anything from sys.argv since we don't want to get
    # parameter conflicts after the simname and work_dir have been decided
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    c = bfps.DNS()
    c.launch(params)
    return None

if __name__ == '__main__':
    main()

