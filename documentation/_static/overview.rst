Overview and Tutorial
=====================

At the moment, this code is meant to run pseudospectral DNS of
Navier-Stokes, using FFTW 3, and to integrate particle trajectories in
the resulting fields.
I'm trying to write it as general as possible, so that it can be
expanded in the future; it remains to be seen how well this will work.

The Navier-Stokes solver has been extensively tested (tests are included
in the repository), and it is working as expected. Parameters and
statistics are stored in HDF5 format, together with code information,
so simulation data should be "future proof".

Users of this code are expected to either use `NavierStokes` objects
directly, or construct their own class that inherits this class.
The way I use it is to I inherit and add custom statistics as necessary; I
also have private C++ code that can get added and used when needed.
I plan on adding documentation on the procedure as when other people
show interest in using the code, or when time permits.


