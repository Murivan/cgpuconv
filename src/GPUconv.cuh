// GPU/CPU Convolution engine
//  GPUconv.cuh
//  GPUconv
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-02-27.
//

#pragma once
#ifdef __cplusplus
extern "C" {
#endif 
	float GPUconv(float*, int, float*, float*, int, float*, float*, int);
#ifdef __cplusplus
}
#endif