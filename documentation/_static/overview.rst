Overview and Tutorial
=====================

Equations
---------

The code uses a fairly standard pseudo-spectral algorithm to solve fluid
equations.
The incompressible Navier Stokes equations in velocity form are as
follows:

.. math::

    \partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} =
    - \nabla p + \nu \Delta \mathbf{u} + \mathbf{f}

In fact, the code solves the vorticity formulation of these equations:

.. math::
    \partial_t \mathbf{\omega} +
    \mathbf{u} \cdot \nabla \mathbf{\omega} =
    \mathbf{\omega} \cdot \nabla \mathbf{u} +
    \nu \Delta \mathbf{\omega} + \nabla \times \mathbf{f}

Tutorial
--------

