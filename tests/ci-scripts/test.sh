#!/bin/bash

# print command
set -x
# stops when fails
set -e

# Init
export destdir=$(pwd)"/ci-installdir"
export pythonbin=/home/ubuntu/anaconda3/bin/python3
export bfpspythonpath=$destdir/lib/python3.6/site-packages/
export PYTHONPATH=:$bfpspythonpath$PYTHONPATH
export PATH=$destdir/bin/:/home/ubuntu/hdf5/install/bin/:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/hdf5/install/lib/:/home/ubuntu/fftw/install/lib/

echo "destdir = $destdir"
echo "pythonbin = $pythonbin"
echo "bfpspythonpath = $bfpspythonpath"

# Remove possible previous installation
if [[ -d $destdir ]] ; then
    rm -rf $destdir ;
fi

# Create install path
if [[ ! -d $bfpspythonpath ]] ; then
    mkdir -p $bfpspythonpath ;
fi

# Build
$pythonbin setup.py compile_library --timing-output 1
# Install
$pythonbin setup.py install --prefix=$destdir

# Test
ls $destdir
ls $destdir/bin/

$pythonbin $destdir/bin/bfps.test_NSVEparticles

# Clean
if [[ -d $destdir ]] ; then
    rm -rf $destdir ;
fi

