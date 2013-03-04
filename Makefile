################################################################################
#
# Build script for project
# Created by Davide Andrea Mauro
# Last Edited by Davide Andrea Mauro on 2013-02-27
#
################################################################################

# Add source files here
EXECUTABLE	:= cGPUconv
# Cuda source files (compiled with cudacc)
CUFILES		:= GPUconv.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:=  main.cpp CPUconv.cpp OCLconv.cpp fft_execute.cpp fft_kernelstring.cpp fft_setup.cpp oclUtils.cpp
# Additional libraries needed by the project
USECUFFT        := 1
USELIBSNDFILE	:= 1
USELIBFFTW3		:= 1
USEOPENCL		:= 1
USEPORTAUDIO		:= 1
USEBOOST		:= 1
emu				:= 0
verbose := 1

################################################################################
# Rules and targets

include common.mk