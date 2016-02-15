===========
C++ classes
===========

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

fields need to be padded, so that interpolation is a 1cpu job.

rFFTW_interpolator
------------------

fields are not padded, computation is synchronized across all CPUs.

-----------------
Particle trackers
-----------------

The point of these classes is to solve ODEs.
They need to be coupled to an interpolator.

particles
---------

*work in progress* distributed solver.

rFFTW_particles
---------------

*NOT* distributed.
Particle data is syncrhonized across all CPUs.

