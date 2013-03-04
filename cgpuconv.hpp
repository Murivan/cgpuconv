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
// includes, project
#include <sndfile.h>
#include <portaudio.h>
#include <boost/program_options.hpp>
#include <GPUconv.cuh>
#include <CPUconv.hpp>
#include <OCLconv.hpp>

namespace po= boost::program_options;
using namespace std;
