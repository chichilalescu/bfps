fluid_solver<float> *fs;
fs = new fluid_solver<float>(32, 32, 32);
DEBUG_MSG("fluid_solver object created\n");

DEBUG_MSG("nu = %g\n", fs->nu);
fs->cd->read(
        "Kdata0",
        (void*)fs->cvorticity);
fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
fs->force_divfree(fs->cvorticity);
fs->symmetrize(fs->cvorticity, 3);
DEBUG_MSG("field read\n");
DEBUG_MSG("######### %d %g\n", fs->iteration, fs->correl_vec(fs->cvorticity, fs->cvorticity));
for (int t = 0; t < 8; t++)
{
    fs->step(0.01);
    DEBUG_MSG("######### %d %g\n", fs->iteration, fs->correl_vec(fs->cvorticity, fs->cvorticity));
}

delete fs;
DEBUG_MSG("fluid_solver object deleted\n");

