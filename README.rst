================================
Big Fluid and Particle Simulator
================================

In brief, this code runs pseudospectral direct numerical simulations
(DNS) of the incompressible Navier-Stokes equations, using FFTW 3, and
it can integrate particle trajectories in the resulting fields.

The Navier-Stokes solver has been extensively tested (tests are included
in the repository), and it is working as expected.
Parameters and statistics are stored in HDF5 format, together with code
information, so simulation data should be "future proof" --- suggestions
of possible improvements to the current approach are always welcome.

The primary aim of bfps is to reduce the time spent on setting up and
baby sitting DNS, as well as simplify the analysis of the generated
data.
The wish is that this Python package provides an easy and general way
of constructing efficient specialized DNS C++ codes for different
turbulence problems encountered in research.
At the same time, the package should provide a unified way of
postprocessing, and accessing the postprocessing results.
The code therefore consists of two main parts: the pure C++ code, a set
of loosely related "building blocks", and the Python code, which can
generate C++ code using the pure classes, but with a significant degree
of flexibility.

The code user is expected to write a small python script that will
properly define the DNS they are interested in running.
That code will generate an executable that can then be run directly on
the user's machine, or submitted to a queue on a cluster.


.. _sec-installation:

------------
Installation
------------

So far, the code has been run on laptops, desktops, and a couple of
clusters (biggest run so far was 1536^3 on 16 nodes of 32 cores each,
with about 11 seconds per time step, for a simple incompressible
Navier-Stokes problem).
Postprocessing data may not be very computationally intensive, depending
on the amount of data involved.

**Postprocessing only**

Use a console; navigate to the ``bfps`` folder, and type:

.. code:: bash

    python setup.py install

(add `--user` or `sudo` as appropriate).
`setup.py` should tell you about the various packages you need.

**Full installation**

If you want to run simulations on the machine where you're installing,
you will need to call `build` before installing.
Your machine needs to have an MPI compiler installed, the HDF5 C library
and FFTW >= 3.4.
The file `machine_settings_py.py` should be modified
appropriately for your machine (otherwise the `build` command will most
likely fail).
This file will be copied the first time you run `setup.py` into
`$HOME/.config/bfps/machine_settings.py`, **where it will be imported from
afterwards** --- any future edits **must** be made to the new file.
You may, obviously, edit it afterwards and rerun the build command as
needed.

.. code:: bash

    python setup.py build
    python setup.py install

-------------
Documentation
-------------

While the code is not fully documented yet, basic information is already
available, and it is recommended that you generate the manual and go
through it carefully.
Please don't be shy about asking for specific improvements to the
current text.
In order to generate the manual, navigate to the repository folder, and
execute the following commands:

.. code:: bash

    cd documentation
    make latexpdf

Optionally, html documentation can be generated instead if needed, just
type ``make html`` instead of ``make latexpdf``.

--------
Comments
--------

* particles: initialization of multistep solvers is done with lower
  order methods, so direct convergence tests will fail.

* Code is used mainly with Python 3.4 and 3.5.
  In principle it should be easy to maintain compatibility with Python
  2.7.x, but as of `bfps 1.8` this is no longer a main concern.

