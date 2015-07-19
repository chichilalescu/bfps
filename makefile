########################################################################
#
#  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: Cristian.Lalescu@ds.mpg.de
#
########################################################################

# environment variables:
#
# FFTW_INCLUDE = location of fftw3.h
# FFTW_LIB = location of libfftw3 etc
# LOCAL_INCLUDE = location of mpi.h
# LOCAL_LIB = location of libmpi



MPICXX  = mpicxx
LINKER  = mpicxx
DEFINES = #-DNDEBUG
CFLAGS  = -Wall \
		  -O2 \
		  -g \
		  -ffast-math \
		  ${LOCAL_INCLUDE} \
		  ${FFTW_INCLUDE}
		  #-pg \
		  #-finstrument-functions

LIBS = ${FFTW_LIB} \
	   -lfftw3_mpi \
	   -lfftw3 \
	   -lfftw3f_mpi \
	   -lfftw3f

COMPILER_VERSION := $(shell ${MPICXX} --version)

ifneq (,$(findstring ICC,$(COMPILER_VERSION)))
	# using intel compiler
	# advice from
	# https://software.intel.com/en-us/forums/topic/298872
	# always link against both libimf and libm
    LIBS += -limf \
			-lm
else
    # not using intel compiler
endif

base_files := \
	field_descriptor \
	fftw_tools \
	Morton_shuffler \
	p3DFFT_to_iR \
	vector_field \
	fluid_solver_base \
	fluid_solver \
	slab_field_particles \
	tracers \
	spline_n1 \
	spline_n2 \
	spline_n3

#headers := $(patsubst %, ./src/%.hpp, ${base_files})
src := $(patsubst %, ./src/%.cpp, ${base_files})
obj := $(patsubst %, ./obj/%.o, ${base_files})

.PRECIOUS: ./obj/%.o

./obj/%.o: ./src/%.cpp
	${MPICXX} ${DEFINES} \
		${CFLAGS} \
		-c $^ -o $@

base: ${obj}
	ar rcs ./lib/libbfps.a $^

%.elf: ${obj} ./obj/%.o
	${LINKER} \
		$^ \
		-o $@ \
		${LIBS} \
		-L./lib/ \
		-lbfps \
		${NULL}

clean:
	rm -f ./obj/*.o
	rm -f *.elf

