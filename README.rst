================================
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

Users of this code are expected to either use `NavierStokes` objects
directly, or construct their own class that inherits this class.
The way I use it is to I inherit and add custom statistics as necessary; I
also have private C++ code that can get added and used when needed.
I plan on adding documentation on the procedure as when other people
show interest in using the code, or when time permits.


.. _sec-installation:

------------
Installation
------------

**Postprocessing only**

.. code:: bash

    python setup.py install

(add `--user` or `sudo` as appropriate).
`setup.py` should tell you about the various packages you need.

**Full installation**

If you want to run simulations on the machine where you're installing,
you will need to call `build` before installing.
Before executing any command, please modify `machine_settings_py.py`
appropriately for your machine (otherwise the `build` command will most
likely fail).
This file will be copied the first time you run `setup.py` into
`$HOME/.config/bfps/machine_settings.py`, where it will be imported from
afterwards.
You may, obviously, edit it afterwards and rerun the build command as
needed.

.. code:: bash

    python setup.py build
    python setup.py install

Also, in order to run the C++ code you need to have an MPI compiler
installed, the HDF5 C library as well as FFTW 3 (at least 3.3 I think).


--------
Comments
--------

* particles: initialization of multistep solvers is done with lower
  order methods, so don't be surprised if direct convergence tests fail.

* I am using this code mainly with Python 3.4, but Python 2.7
  compatibility should be kept since mayavi (well, vtk actually) only
  works on Python 2.

