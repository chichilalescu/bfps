Big Fluid and Particle Simulator
================================

At the moment, this code is meant to run pseudospectral DNS of
Navier-Stokes, using FFTW 3, and to integrate particle trajectories in
the resulting fields.
I'm trying to write it as general as possible, so that it can be
expanded in the future; it remains to be seen how well this will work.

The Navier-Stokes solver has been extensively tested (tests are included
in the repository), and it is working as expected. Parameters and
statistics are stored in HDF5 format, together with code information,
so simulation data should be "future proof".

Comments
--------

* particles: initialization of multistep solvers is done with lower
  order methods, so don't be surprised if naive convergence tests fail.

TODO
----

* store time averaged stats in .h5 file

* store nonlinear term for particles

* add different numerical method, for testing purposes

* test involving hydrodynamic similarity

* test anisotropic grids

* test non-cubic domains

* use HDF5 io for fields

* use FFTW wisdom with some host dependent cache

* try to make code more memory efficient

* make code py3 compatible

