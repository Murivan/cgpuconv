// GPU/CPU Convolution engine
//  cgpuconv.hpp
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-02-27.
//

    // Do not include headers multiple times
#pragma once

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

// includes, libraries
#include <sndfile.h>
#include <portaudio.h>
#include <boost/program_options.hpp>


// includes, project
#include <GPUconv.cuh>
#include <CPUconv.hpp>
#include <OCLconv.hpp>

//Define for PortAudio types
/* This is the current sample format: That is 44.1kHz signed int 16 bits*/
#define SAMPLE_RATE  (44100)
#define PA_SAMPLE_TYPE  paInt16
typedef short SAMPLE;
#define SAMPLE_SIZE (sizeof(SAMPLE))
#define SAMPLE_SILENCE  (0)
#define SAMPLE_PER_BUFFER  (320)
#define INPUT_CHANNELS  (1)
#define OUTPUT_CHANNELS  (2)
#define PRINTF_S_FORMAT "%d"

namespace po= boost::program_options;
using namespace std;
