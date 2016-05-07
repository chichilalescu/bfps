===========
C++ classes
===========

----------------
Field operations
----------------

-----
field
-----

Class that should handle all field operations.
I/O is performed with HDF5 (maybe the binary I/O should be migrated here
for flexibility).
Can perform real space statistics.

------
kspace
------

Class that describes Fourier space.
Can compute spectra.
Should compute various Fourier filters, and should handle any
filter-based dealiasing as well.

-------------
Fluid Solvers
-------------

-------------
Interpolators
-------------

The point of these classes is to take care of interpolation.
Given the number of neighbours used, and the type of interpolation, they
compute the polynomials, and they compute the big sum.
They need to have at least the following public methods:

.. code:: cpp

    /* map real locations to grid coordinates */
    void get_grid_coordinates(
            const int nparticles,
            const int pdimension,
            const double *__restrict__ x,
            int *__restrict__ xg,
            double *__restrict__ xx);
    /* interpolate field at an array of locations */
    void sample(
            const int nparticles,
            const int pdimension,
            const double *__restrict__ x,
            double *__restrict__ y,
            const int *deriv = NULL);
    /* interpolate 1 point */
    void operator()(
            const int *__restrict__ xg,
            const double *__restrict__ xx,
            double *__restrict__ dest,
            const int *deriv = NULL);

interpolator
------------

Fields need to be padded, so that interpolation is a 1cpu job.
While slow, codes using this interpolator will always work.

rFFTW_interpolator
------------------

Fields are not padded, computation is synchronized across different processes.
Since this avoids the padding step, it's ridiculously faster for small
numbers of particles.

-----------------
Particle trackers
-----------------

The point of these classes is to solve ODEs.
They need to be coupled to an interpolator.

particles
---------

Simple class, particle state is synchronized across all processes.
It can work either with ``interpolator`` or ``rFFTW_interpolator``.

distributed_particles
---------------------

Works **only** with ``interpolator``.
Particles are split among different processes, and redistributed after every
time step.

rFFTW_distributed_particles
---------------------------

Works **only** with ``rFFTW_interpolator``, and **only** if the interpolation
kernel is at least as big as the `z` size of the local field slab.
Particles are split among different domains, and redistributed after
every time step.
Each process has 3 domains: particles for which interpolation needs to
be synchronized with the lower `z` process, particles for which the
interpolation is local, and particles for which the interpolation needs
to be synchronized with the higher `z` process.

