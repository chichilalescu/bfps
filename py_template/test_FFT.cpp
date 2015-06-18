fluid_solver<float> *fs;
fs = new fluid_solver<float>(32, 32, 32);
DEBUG_MSG("fluid_solver object created\n");

vector_field<float> cv(fs->cd, fs->cvorticity);
vector_field<float> rv(fs->cd, fs->rvorticity);

fs->cd->read(
        "Kdata0",
        (void*)fs->cvorticity);
fftwf_execute(*(fftwf_plan*)fs->c2r_vorticity);
//rv*(1. / (fs->rd->sizes[0]*fs->rd->sizes[1]*fs->rd->sizes[2]));
fftwf_execute(*(fftwf_plan*)fs->r2c_vorticity);
cv = cv*(1. / (fs->rd->sizes[0]*fs->rd->sizes[1]*fs->rd->sizes[2]));
fs->cd->write(
        "Kdata1",
        (void*)fs->cvorticity);

DEBUG_MSG("full size is %ld\n", fs->rd->full_size);

delete fs;
DEBUG_MSG("fluid_solver object deleted\n");

