import bfps

def main():
    c = bfps.NavierStokes()
    c.fill_up_fluid_code()
    c.parameters['nparticles'] = 100
    c.add_interpolator(
            name = 'interp',
            class_name = 'rFFTW_interpolator')
    c.add_particles(
            interpolator = 'interp',
            class_name = 'rFFTW_distributed_particles')
    # now add tensor interpolation hack
    c.particle_stat_src += """
        {
        field<float, FFTW, THREE> *vec_field;
        field<float, FFTW, THREExTHREE> *vec_gradient;
        kspace<FFTW, SMOOTH> *kk_smooth;

        kk_smooth = new kspace<FFTW, SMOOTH>(
                tmp_vec_field->clayout,
                fs->dkx, fs->dky, fs->dkz);
        vec_field = new field<float, FFTW, THREE>(
                nx, ny, nz,
                MPI_COMM_WORLD,
                {1});

        vec_gradient = new field<float, FFTW, THREExTHREE>(
                nx, ny, nz,
                MPI_COMM_WORLD,
                {1});

        *vec_field = fs->rvelocity;
        vec_field->dft();
        compute_gradient(
            kk_smooth,
            vec_field,
            vec_gradient);
        vec_gradient->ift();
        delete vec_field;
        delete vec_gradient;
        }
                           """.format(c.fftw_plan_rigor)
    c.finalize_code()
    c.write_src()
    return None

if __name__ == '__main__':
    main()

