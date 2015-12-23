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


===========
Development
===========

`bfps` is using `git` for version tracking (see https://git-scm.com/
for description).
The branching model described at
http://nvie.com/posts/a-successful-git-branching-model/ should be
adhered to as strictly as possible.

The usable `VERSION` number will be constructed according to semantic
versioning rules (see http://semver.org/), `MAJOR.MINOR.PATCH`.
In principle, if you use `bfps` and you've created children of the
`NavierStokes` class, you should not need to rewrite your code unless
the `MAJOR` version changes.

Versioning guidelines
---------------------

There are 2 main branches, `master` and `develop`.
`setup.py` will call `git` to read in the `VERSION`: it will get the
latest available tag.
If the active branch name contains either of the strings `develop`,
`feature` or `bugfix`, then the full output of `git describe --tags`
will be used;
otherwise, only the string before the dash (if a dash exists) will be
used.

At the moment the following rules seem adequate.
Once I get feedback from someone who actually knows how to handle bigger
projects, these may change.

1. New features are worked on in branches forked from `develop`, with
   the branch name of the form `feature/bla`.
   Feature branches are merged back into `develop` only after all tests
   pass.
2. Whenever the `develop` branch is merged into `master`, the last
   commit on the `develop` branch is tagged with `MAJOR.MINOR`.
3. Whenever a bug is discovered in version X.Y, a new branch called `vX.Y`
   is forked from the corresponding `master` merge point.
   A new bugfix branch is forked from `vX.Y`, the bug is fixed, and then
   this bugfix branch is merged into all affected branches (this includes
   `vX.Y`).
   After merging, the respective merge points are tagged adequately (`vX.Y`
   gets the tag X.Y.1).
4. Whenever a bug is discovered in version X.Y.Z, a bugfix branch is
   forked from `vX.Y`, and then rule 3 is adapted accordingly.

