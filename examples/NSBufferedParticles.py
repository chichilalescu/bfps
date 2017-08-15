import bfps
import argparse
import sys

class NSBufferedParticles(bfps.NavierStokes):
    """
        Another example.
        This class behaves identically to NavierStokes, except that it uses a
        buffered interpolator, and the corresponding distributed_particles class.
    """
    standard_names = ['NSBP',
                      'NSBP-single',
                      'NSBP-double']
    def launch(
            self,
            args = [],
            noparticles = False,
            **kwargs):
        self.name = 'NSBufferedParticles-v' + bfps.__version__
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        if noparticles:
            opt.nparticles = 0
        elif type(opt.nparticles) == int:
            if opt.nparticles > 0:
                self.name += '-particles'
                self.add_3D_rFFTW_field(
                        name = 'rFFTW_acc')
                self.add_interpolator(
                        name = 'cubic_spline',
                        neighbours = opt.neighbours,
                        smoothness = opt.smoothness,
                        class_name = 'interpolator')
                self.add_particles(
                        integration_steps = [4],
                        interpolator = 'cubic_spline',
                        acc_name = 'rFFTW_acc',
                        class_name = 'distributed_particles')
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'NSBufferedParticles')
    parser.add_argument(
            '-v', '--version',
            action = 'version',
            version = '%(prog)s ' + bfps.__version__)
    c = NSBufferedParticles(fluid_precision = 'single')
    c.launch(args = sys.argv[1:])

