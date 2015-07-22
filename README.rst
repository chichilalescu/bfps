Big Fluid and Particle Simulator
================================

At the moment, this code is meant to run pseudospectral DNS of
Navier-Stokes, using FFTW 3, and to integrate particle trajectories in
the resulting fields.
I'm trying to write it as general as possible, so that it can be
expanded in the future; it remains to be seen how well this will work.

TODO
----

* multi-step method for particles

* try to make code more memory efficient

* complex field IO should be space efficient (i.e. don't write modes
  that are 0 due to dealiasing scheme)

* make templates work for `double` as well, and python wrappers
  should control precision.
