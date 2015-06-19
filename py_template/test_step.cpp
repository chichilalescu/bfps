fluid_solver<float> *fs;
fs = new fluid_solver<float>(32, 32, 32);
DEBUG_MSG("fluid_solver object created\n");

fs->cd->read(
        "Kdata0",
        (void*)fs->cvorticity);
DEBUG_MSG("field read\n");
DEBUG_MSG("######### %d %g\n", fs->iteration, fs->correl_vec(fs->cvorticity, fs->cvorticity));
fs->step(0.0001);
DEBUG_MSG("######### %d %g\n", fs->iteration, fs->correl_vec(fs->cvorticity, fs->cvorticity));

delete fs;
DEBUG_MSG("fluid_solver object deleted\n");

