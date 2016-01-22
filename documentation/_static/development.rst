===========
Development
===========


---------------------
Versioning guidelines
---------------------

Version tracking for `bfps` is done with `git` (see https://git-scm.com/
for description).
The branching model described at
http://nvie.com/posts/a-successful-git-branching-model/ should be
adhered to as strictly as possible.

The usable `VERSION` number will be constructed according to semantic
versioning rules (see http://semver.org/), `MAJOR.MINOR.PATCH`.
In principle, if you use `bfps` and you've created children of the
`NavierStokes` class, you should not need to rewrite your code unless
the `MAJOR` version changes.

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


------------
Code testing
------------

Testing for `bfps` is done with `tox`.
