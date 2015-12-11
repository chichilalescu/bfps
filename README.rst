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

.. code:: bash
    python setup.py build
    python setup.py install

The `build` command will most likely fail unless you modify
`machine_settings.py` appropriately for your machine.
Also, in order to run the C++ code you need to have an MPI compiler
installed, the HDF5 C library as well as FFTW 3 (at least 3.3 I think).


Comments
--------

* particles: initialization of multistep solvers is done with lower
  order methods, so don't be surprised if direct convergence tests fail.

* I am using this code mainly with Python 3.4, but Python 2.7
  compatibility should be kept since mayavi (well, vtk actually) only
  works on Python 2.

