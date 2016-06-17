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



import bfps

class NSManyParticles(bfps.NavierStokes):
    def specific_parser_arguments(
            self,
            parser):
        bfps.NavierStokes.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--particle-class',
                default = 'rFFTW_distributed_particles',
                dest = 'particle_class',
                type = str)
        parser.add_argument(
                '--interpolator-class',
                default = 'rFFTW_interpolator',
                dest = 'interpolator_class',
                type = str)
        parser.add_argument('--neighbours',
                type = int,
                dest = 'neighbours',
                default = 3)
        parser.add_argument('--smoothness',
                type = int,
                dest = 'smoothness',
                default = 2)
        return None
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        if type(opt.nparticles) == int:
            if opt.nparticles > 0:
                self.add_3D_rFFTW_field(
                        name = 'rFFTW_acc')
                interp_list = []
                for n in range(1, opt.neighbours):
                    interp_list.append('Lagrange_n{0}'.format(n))
                    self.add_interpolator(
                            interp_type = 'Lagrange',
                            name = interp_list[-1],
                            neighbours = n,
                            class_name =  opt.interpolator_class)
                    for m in range(1, opt.smoothness):
                        interp_list.append('spline_n{0}m{1}'.format(n, m))
                        self.add_interpolator(
                                interp_type = 'spline',
                                name = interp_list[-1],
                                neighbours = n,
                                smoothness = m,
                                class_name =  opt.interpolator_class)
                self.add_particles(
                        integration_steps = 2,
                        interpolator = interp_list,
                        acc_name = 'rFFTW_acc',
                        class_name = opt.particle_class)
                self.add_particles(
                        integration_steps = 4,
                        interpolator = interp_list,
                        acc_name = 'rFFTW_acc',
                        class_name = opt.particle_class)
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None

