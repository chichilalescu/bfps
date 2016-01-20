import os

import sys
import numpy as np
import bfps

class Launcher:
    def __init__(
            self,
            data_dir = './'):
        self.parser = bfps.get_parser(
                bfps.NavierStokes,
                work_dir = os.path.realpath(data_dir))
        self.parser.add_argument(
                '--QR-stats',
                action = 'store_true',
                dest = 'QR_stats')
        self.parser.add_argument(
                '--kMeta',
                type = float,
                dest = 'kMeta',
                default = 2.0)
        self.parser.add_argument(
                '--dtfactor',
                type = float,
                dest = 'dtfactor',
                default = 0.5,
                help = 'dt is computed as DTFACTOR / N')
        self.parser.add_argument(
                '--environment',
                type = str,
                dest = 'environment',
                default = '')
        self.parser.add_argument(
                '--src-simname',
                type = str,
                dest = 'src_simname',
                default = '')
        self.parser.add_argument(
                '--src-iteration',
                type = int,
                dest = 'src_iteration',
                default = 0)
        self.data_dir = data_dir
        self.base_class = bfps.NavierStokes
        return None
    def __call__(
            self,
            args = None):
        opt = self.parser.parse_args(args)
        if opt.environment != '':
            bfps.host_info['environment'] = opt.environment
        opt.nx = opt.n
        opt.ny = opt.n
        opt.nz = opt.n
        opt.work_dir = os.path.join(
                os.path.realpath(opt.work_dir),
                'N{0:0>4}'.format(opt.n))
        c = self.base_class(
                fluid_precision = opt.precision,
                simname = opt.simname,
                QR_stats_on = opt.QR_stats)
        c.pars_from_namespace(opt)
        # with the default Lundgren forcing, I can estimate the dissipation
        # with nondefault forcing, figure out the amplitude for this viscosity
        # yourself
        c.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
        c.parameters['dt'] = (opt.dtfactor / opt.n)
        if ((c.parameters['niter_todo'] % c.parameters['niter_out']) != 0):
            c.parameters['niter_out'] = c.parameters['niter_todo']
        if c.QR_stats_on:
            # max_Q_estimate and max_R_estimate are just used for the 2D pdf
            # therefore I just want them to be small multiples of mean trS2
            # I'm already estimating the dissipation with kMeta...
            meantrS2 = (opt.n//2 / opt.kMeta)**4 * c.parameters['nu']**2
            c.parameters['max_Q_estimate'] = meantrS2
            c.parameters['max_R_estimate'] = .4*meantrS2**1.5
        c.fill_up_fluid_code()
        c.finalize_code()
        c.write_src()
        c.set_host_info(bfps.host_info)
        if opt.run:
            if not os.path.exists(os.path.join(c.work_dir, c.simname + '.h5')):
                c.write_par()
                if c.parameters['nparticles'] > 0:
                    if opt.particle_rand_seed != 0:
                        rseed = opt.particle_rand_seed
                    else:
                        rseed = None
                    data = c.generate_tracer_state(species = 0, rseed = rseed)
                    for s in range(1, c.particle_species):
                        c.generate_tracer_state(species = s, data = data)
                init_condition_file = os.path.join(
                        c.work_dir,
                        c.simname + '_cvorticity_i{0:0>5x}'.format(0))
                if not os.path.exists(init_condition_file):
                    if len(opt.src_simname) > 0:
                        src_file = os.path.join(
                                c.work_dir,
                                opt.src_simname + '_cvorticity_i{0:0>5x}'.format(opt.src_iteration))
                        os.symlink(src_file, init_condition_file)
                    else:
                       c.generate_vector_field(
                               write_to_file = True,
                               spectra_slope = 2.0,
                               amplitude = 0.25)
            c.run(ncpu = opt.ncpu,
                  njobs = opt.njobs)
        return c


