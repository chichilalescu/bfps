#!/bin/bash

# print command
set -x
# stops when fails
set -e

# Init
source ~/.bashrc
export destdir=/tmp/bfpsinstall
export pythonbin=/home/ubuntu/anaconda3/bin/python3
export bfpspythonpath=$destdir/lib/python3.6/site-packages/
export PYTHONPATH=:$bfpspythonpath$PYTHONPATH
export PATH=$destdir/bin/:$PATH

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

$pythonbin $destdir/bin/test_bfps_NSVEparticles.py

# Clean
if [[ -d $destdir ]] ; then
    rm -rf $destdir ;
fi

