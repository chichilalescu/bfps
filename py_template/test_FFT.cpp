fluid_solver<float> *fs;
fs = new fluid_solver<float>(32, 32, 32);
DEBUG_MSG("fluid_solver object created\n");

fs->fc->read(
        "Kdata0",
        (void*)fs->cvorticity);
fftwf_execute(*(fftwf_plan*)fs->c2r_vorticity);
fftwf_execute(*(fftwf_plan*)fs->r2c_vorticity);
fs->fc->write(
        "Kdata1",
        (void*)fs->cvorticity);

delete fs;
DEBUG_MSG("fluid_solver object deleted\n");

