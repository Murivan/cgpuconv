// GPU/CPU Convolution engine
//  OCLComplexPwiseMul.cl
//  OCLconv
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-02-27.
//

// OpenCL Kernel Function for element by element vector Complex Pointwise Multiplication
//__kernel void ComplexPointwiseMul(__global  float* a, __global  float* b, __global const float* c, __global const float* d, int iNumElements)
//{
//    // get index into global data array
//    int iGID = get_global_id(0);
//
//    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
//    if (iGID >= iNumElements)
//    {   
//        return; 
//    }
//    a[iGID] = a[iGID] * c[iGID] - b[iGID] * d[iGID] ;
//    b[iGID] = a[iGID] * d[iGID] + b[iGID] * c[iGID] ;
//}
//
__kernel void ComplexPointwiseMul(__global float* a, __global float* b, __global const float* c, __global const float* d, int size)
{
        const int numThreads = get_local_size(0) * get_num_groups(0);
        for (int i = get_global_id(0); i < size; i += numThreads){
        	float k=a[i];
      		  a[i] = (a[i] * c[i]) - (b[i] * d[i]);
        	  b[i]= (k * d[i]) + (b[i] * c[i]);
        	  }
}