#######################################################################
#                                                                     #
#  Copyright 2015 Max Planck Institute                                #
#                 for Dynamics and Self-Organization                  #
#                                                                     #
#  This file is part of bfps.                                         #
#                                                                     #
#  bfps is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published  #
#  by the Free Software Foundation, either version 3 of the License,  #
#  or (at your option) any later version.                             #
#                                                                     #
#  bfps is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of     #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
#  GNU General Public License for more details.                       #
#                                                                     #
#  You should have received a copy of the GNU General Public License  #
#  along with bfps.  If not, see <http://www.gnu.org/licenses/>       #
#                                                                     #
# Contact: Cristian.Lalescu@ds.mpg.de                                 #
#                                                                     #
#######################################################################



import os
import sys
import bfps
import numpy as np

class NS0SliceParticles(bfps.NavierStokes):
    """
        Example of how bfps is envisioned to be used.
        Standard NavierStokes class is inherited, and then new functionality
        added on top.
        In particular, this class will a DNS with particles starting on a square
        grid in the z=0 slice of the field.
    """
    standard_names = ['NS0SP',
                      'NS0SP-single',
                      'NS0SP-double']
    def __init__(
            self,
            name = 'NS0SliceParticles-v' + bfps.__version__,
            **kwargs):
        bfps.NavierStokes.__init__(
                self,
                name = name,
                **kwargs)
        return None
    def specific_parser_arguments(
            self,
            parser):
        bfps.NavierStokes.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--pcloudX',
                type = float,
                dest = 'pcloudX',
                default = 0.0)
        parser.add_argument(
                '--pcloudY',
                type = float,
                dest = 'pcloudY',
                default = 0.0)
        return None
    def launch_jobs(
            self,
            opt = None):
        if not os.path.exists(os.path.join(self.work_dir, self.simname + '.h5')):
            particle_initial_condition = None
            if self.parameters['nparticles'] > 0:
                # the extra dimension of 1 is because I want
                # a single chunk of particles.
                particle_initial_condition = np.zeros(
                        (1,
                         self.parameters['nparticles'],
                         self.parameters['nparticles'],
                         3),
                        dtype = np.float64)
                xvals = (opt.pcloudX +
                         np.linspace(-opt.particle_cloud_size/2,
                                      opt.particle_cloud_size/2,
                                      self.parameters['nparticles']))
                yvals = (opt.pcloudY +
                         np.linspace(-opt.particle_cloud_size/2,
                                      opt.particle_cloud_size/2,
                                      self.parameters['nparticles']))
                particle_initial_condition[..., 0] = xvals[None, None, :]
                particle_initial_condition[..., 1] = yvals[None, :, None]
            self.write_par(
                    particle_ic = particle_initial_condition)
            if self.parameters['nparticles'] > 0:
                data = self.generate_tracer_state(
                        species = 0,
                        rseed = opt.particle_rand_seed,
                        data = particle_initial_condition)
            init_condition_file = os.path.join(
                    self.work_dir,
                    self.simname + '_cvorticity_i{0:0>5x}'.format(0))
            if not os.path.exists(init_condition_file):
                if len(opt.src_simname) > 0:
                    src_file = os.path.join(
                            os.path.realpath(opt.src_work_dir),
                            opt.src_simname + '_cvorticity_i{0:0>5x}'.format(opt.src_iteration))
                    os.symlink(src_file, init_condition_file)
                else:
                   self.generate_vector_field(
                           write_to_file = True,
                           spectra_slope = 2.0,
                           amplitude = 0.05)
        self.run(
                ncpu = opt.ncpu,
                njobs = opt.njobs,
                hours = opt.minutes // 60,
                minutes = opt.minutes % 60)
        return None

def main():
    c = NS0SliceParticles()
    c.launch(args = sys.argv[1:])
    return None

if __name__ == '__main__':
    main()

