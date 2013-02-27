// GPU/CPU Convolution engine
//  OCLconv.hpp
//  OCLconv
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-02-27.
//

#pragma once
#ifdef __cplusplus
extern "C" {
#endif 
	float OCLconv(float*, int, float*, float*, int, float*, float*, int, char*);
#ifdef __cplusplus
}
#endif 