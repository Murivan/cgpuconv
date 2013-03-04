// GPU/CPU Convolution engine
//  GPUconv.cu
//  GPUconv
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-02-27.
//

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cufft.h>
//#include <cutil_inline.h>
//#include <shrQATest.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

#include <GPUconv.cuh>


static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
static __global__ void ComplexPointwiseMul(cufftComplex*, const cufftComplex* , int size);

float GPUconv(float* input, int SIGNAL_SIZE, float* filtersx, float* filterdx, int FILTER_KERNEL_SIZE, float* outputsx, float* outputdx, int direct) 
{
	//Look for CUDA capable Devices
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		printf("There is no device supporting CUDA\n");
		return -1.0f;
	}
	//cudaDeviceReset();

	//Pick the best one
	//int rId=cutGetMaxGflopsDeviceId();
	int rId=gpuGetMaxGflopsDeviceId();
	cudaSetDevice(rId);
	//Pick properties used for block and grid
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, rId);

	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	//printf("Free/Total %lu/%lu %u %%\n",free_mem, total_mem, (unsigned int) ((free_mem*100.0)/total_mem));	
	//cudaError error = cudaThreadSetLimit(cudaLimitMallocHeapSize, (size_t) (0.8 * free_mem));

	if(direct==1){
		//Work on it! Caution Memory limit
		int required_mem=3*((SIGNAL_SIZE+FILTER_KERNEL_SIZE-1)*sizeof(cufftComplex));
		//printf("Required Memory: %d.\n", required_mem);
		if (required_mem>=(free_mem*0.8)){
			printf("Insufficient memory on device. Required %d, Available %lu.\n", required_mem, free_mem);
			return -1.0f;
		}

		int new_size=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
		int mem_size=sizeof(cufftComplex) *new_size;
		printf("Entering Direct Mode.\n");

		//First thing to do: PAD!
		cufftComplex* h_signal = (cufftComplex*)malloc(mem_size);
		// Initalize the memory for the signal
		for (int i = 0; i < new_size; i++) {
			if (i<SIGNAL_SIZE){
				h_signal[i].x = input[i];
			}
			else{
				h_signal[i].x = 0.0f;
			}
			h_signal[i].y = 0.0f;
		}

		cufftComplex* h_filter_kernels[2];
		for (int k=0; k<2;k++){
			h_filter_kernels[k]=(cufftComplex*)malloc(mem_size);
		}
		// Initalize the memory for the filter
		for (int i = 0; i < new_size; i++) {
			if(i<FILTER_KERNEL_SIZE){
				h_filter_kernels[0][i].x = filtersx[i];
				h_filter_kernels[1][i].x = filterdx[i];
			}
			else{
				h_filter_kernels[0][i].x = 0.0f;
				h_filter_kernels[1][i].x = 0.0f;
			}
			h_filter_kernels[0][i].y = 0.0f;
			h_filter_kernels[1][i].y = 0.0f;
		}


		// CUFFT plan
		cufftHandle plan;	
		int window=new_size;
		//cufftSafeCall(cufftPlan1d(&plan, window, CUFFT_C2C, (new_size/window)));
		cufftPlan1d(&plan, window, CUFFT_C2C, (new_size/window));

		// Allocate device memory for signal
		cufftComplex* d_signal;
		//cutilSafeCall(cudaMalloc((void**)&d_signal, mem_size));
		cudaMalloc((void**)&d_signal, mem_size);

		// Copy host memory to device
		//cutilSafeCall(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));
		cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);
		//   printf("Device Memory allocated for Signal.\n");


		// Allocate device memory for filter kernel
		cufftComplex* d_filter_kernels[2];
		for (int i=0; i<2;i++){
			//cutilSafeCall(cudaMalloc((void**)&d_filter_kernels[i], mem_size));
			//// Copy host memory to device
			//cutilSafeCall(cudaMemcpy(d_filter_kernels[i], h_filter_kernels[i], mem_size, cudaMemcpyHostToDevice));
			cudaMalloc((void**)&d_filter_kernels[i], mem_size);
			// Copy host memory to device
			cudaMemcpy(d_filter_kernels[i], h_filter_kernels[i], mem_size, cudaMemcpyHostToDevice);
		}
		// printf("Device Memory allocated for Filters.\n");


		//	printf("Transforming signal cufftExecC2C\n");
		//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));
		cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

		for (int i=0; i<2;i++){
			//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_FORWARD));
			cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_FORWARD);

		}
		cudaThreadSynchronize();


		// Multiply the coefficients together and normalize the result
		int block_size = 256;//deviceProp.maxThreadsPerBlock; // 
		int grid_size = new_size/256 ;//deviceProp.warpSize; //

		for (int i=0; i<2;i++)
			ComplexPointwiseMul<<<grid_size, block_size>>>(d_filter_kernels[i], d_signal, new_size);
		cudaThreadSynchronize();
		// Check if kernel execution generated and error
		//cutilCheckMsg("Kernel execution failed [ ComplexPointwiseMul ]");


		// Transform signal back
		for (int i=0; i<2;i++){
			//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_INVERSE));
			cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_INVERSE);

		}
		cudaThreadSynchronize();

		// Copy device memory to host
		cufftComplex* h_convolved_signal[2];
		for (int i=0; i<2;i++){
			h_convolved_signal[i]= (cufftComplex*)malloc(mem_size);
			//cutilSafeCall(cudaMemcpy(h_convolved_signal[i], d_filter_kernels[i], mem_size, cudaMemcpyDeviceToHost));
			cudaMemcpy(h_convolved_signal[i], d_filter_kernels[i], mem_size, cudaMemcpyDeviceToHost);
		}


		//printf("Writing back.\n");
		//outputsx=(float*)malloc(sizeof(float) * new_size);
		//outputdx=(float*)malloc(sizeof(float) * new_size);

		float maxo[2];
		maxo[0]=0.0f;
		maxo[1]=0.0f;  
		for (int i = 0; i < new_size; i++){
			if (abs(maxo[0])<=abs(h_convolved_signal[0][i].x)) maxo[0]=h_convolved_signal[0][i].x;
			if (abs(maxo[1])<=abs(h_convolved_signal[1][i].x)) maxo[1]=h_convolved_signal[1][i].x;
		}
		float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);
		for (int i = 0; i < new_size; i++) {
			outputsx[i]=(h_convolved_signal[0][i].x/maxot);
			outputdx[i]=(h_convolved_signal[1][i].x/maxot);
		}

		//printf("Freeing resources.\n");
		//Destroy CUFFT context
		//cufftSafeCall(cufftDestroy(plan));
		cufftDestroy(plan);
		
		// cleanup memory
		free(h_signal);
		//cutilSafeCall(cudaFree(d_signal));
		cudaFree(d_signal);
		
		for (int i=0; i<2;i++){
			free(h_filter_kernels[i]);
			free(h_convolved_signal[i]);
			//cutilSafeCall(cudaFree(d_filter_kernels[i]));
			cudaFree(d_filter_kernels[i]);
		}
		cudaDeviceSynchronize();
		//cutilDeviceReset(); 
		cudaDeviceReset(); 
		return new_size;
	}


	if(direct==0){
		//Work on it! Caution Memory limit
		int required_mem=3*((FILTER_KERNEL_SIZE*2)*sizeof(cufftComplex));
		//printf("Required Memory: %d.\n", required_mem);
		if (required_mem>=(free_mem*0.8)){
			printf("Insufficient memory on device. Required %d, Available %lu.\n", required_mem, free_mem);
			return -1.0f;
		}
		printf("Entering Overlap and Save Mode.\n");

		int new_size=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
		int mem_size=sizeof(cufftComplex) *new_size;
		int payload=FILTER_KERNEL_SIZE*2;
		int mem_pay= sizeof(cufftComplex)*payload;


		cufftComplex* h_filter_kernels[2];
		for (int k=0; k<2;k++){
			h_filter_kernels[k]=(cufftComplex*)malloc(mem_pay);
		}
		// Initalize the memory for the filter
		for (int i = 0; i < payload; i++) {
			if(i<FILTER_KERNEL_SIZE){
				h_filter_kernels[0][i].x = filtersx[i];
				h_filter_kernels[1][i].x = filterdx[i];
			}
			else{
				h_filter_kernels[0][i].x = 0.0f;
				h_filter_kernels[1][i].x = 0.0f;
			}
			h_filter_kernels[0][i].y = 0.0f;
			h_filter_kernels[1][i].y = 0.0f;
		}

		// CUFFT plan
		cufftHandle plan;	
		int window=payload;
		//cufftSafeCall(cufftPlan1d(&plan, window, CUFFT_C2C, (payload/window)));
		cufftPlan1d(&plan, window, CUFFT_C2C, (payload/window));
		// Allocate device memory for filter kernel
		cufftComplex* d_filter_kernels[2];
		for (int i=0; i<2;i++){
			//cutilSafeCall(cudaMalloc((void**)&d_filter_kernels[i], mem_pay));
			cudaMalloc((void**)&d_filter_kernels[i], mem_pay);
			// Copy host memory to device
			//cutilSafeCall(cudaMemcpy(d_filter_kernels[i], h_filter_kernels[i], mem_pay, cudaMemcpyHostToDevice));
			cudaMemcpy(d_filter_kernels[i], h_filter_kernels[i], mem_pay, cudaMemcpyHostToDevice);
		}
		// printf("Device Memory allocated for Filters.\n");
		//	printf("Transforming signal cufftExecC2C\n");
		cudaThreadSynchronize();
		for (int i=0; i<2;i++){
			//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_FORWARD));
			cufftExecC2C(plan, (cufftComplex *)d_filter_kernels[i], (cufftComplex *)d_filter_kernels[i], CUFFT_FORWARD);
		}
		cudaThreadSynchronize();


		float maxo[2];
		maxo[0]=0.0f;
		maxo[1]=0.0f; 

		float* tresultsx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)));
		float* tresultdx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)));

		for(int j=0; j<(SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)); j++){ 
			tresultsx[j]=0.0f;
			tresultdx[j]=0.0f;
		}

		//BIG Loop
		for(int k=0; k<SIGNAL_SIZE; (k+=FILTER_KERNEL_SIZE)){
			//printf("%d.\n",k);
			//First thing to do: PAD!
			cufftComplex* h_signal = (cufftComplex*)malloc(mem_pay);
			// Initalize the memory for the signal
			for (int i = 0; i < payload; i++) {
				if (((k+i)<SIGNAL_SIZE)&&(i<(payload/2))){
					h_signal[i].x = input[k+i];
				}
				else{
					h_signal[i].x = 0.0f;
				}
				h_signal[i].y = 0.0f;
			}
			// Allocate device memory for signal
			cufftComplex* d_signal[2];
			for (int i=0; i<2;i++){
				//cutilSafeCall(cudaMalloc((void**)&d_signal[i], mem_pay));
				cudaMalloc((void**)&d_signal[i], mem_pay);
				// Copy host memory to device
				//cutilSafeCall(cudaMemcpy(d_signal[i], h_signal, mem_pay, cudaMemcpyHostToDevice));
				cudaMemcpy(d_signal[i], h_signal, mem_pay, cudaMemcpyHostToDevice);

				//   printf("Device Memory allocated for Signal.\n");
				//	printf("Transforming signal cufftExecC2C\n");
				//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal[i], (cufftComplex *)d_signal[i], CUFFT_FORWARD));
				cufftExecC2C(plan, (cufftComplex *)d_signal[i], (cufftComplex *)d_signal[i], CUFFT_FORWARD);
			}
			cudaThreadSynchronize();
			// Multiply the coefficients together and normalize the result
			int block_size = deviceProp.maxThreadsPerBlock; // 256;//
			int grid_size = deviceProp.warpSize; //new_size/256 ;//

			for (int i=0; i<2;i++){
				ComplexPointwiseMul<<<grid_size, block_size>>>(d_signal[i], d_filter_kernels[i], payload);
				// Check if kernel execution generated and error
				//cutilCheckMsg("Kernel execution failed [ ComplexPointwiseMul ]");
			}
			cudaThreadSynchronize();
			// Transform signal back
			for (int i=0; i<2;i++){
				//cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal[i], (cufftComplex *)d_signal[i], CUFFT_INVERSE));
				cufftExecC2C(plan, (cufftComplex *)d_signal[i], (cufftComplex *)d_signal[i], CUFFT_INVERSE);
			}
			cudaThreadSynchronize();


			// Copy device memory to host
			cufftComplex* h_convolved_signal[2];
			for (int i=0; i<2;i++){
				h_convolved_signal[i]= (cufftComplex*)malloc(mem_pay);
				//cutilSafeCall(cudaMemcpy(h_convolved_signal[i], d_signal[i], mem_pay, cudaMemcpyDeviceToHost));
				cudaMemcpy(h_convolved_signal[i], d_signal[i], mem_pay, cudaMemcpyDeviceToHost);
			}
			//printf("Writing back.\n");
			for (int i=0; i< payload; i++){
				tresultsx[k+i]+=h_convolved_signal[0][i].x;
				tresultdx[k+i]+=h_convolved_signal[1][i].x;
				if (abs(maxo[0])<=abs(tresultsx[k+i])) maxo[0]=tresultsx[k+i];
				if (abs(maxo[1])<=abs(tresultdx[k+i])) maxo[1]=tresultdx[k+i];
			}

			free(h_signal);
			free(h_convolved_signal[0]);
			free(h_convolved_signal[1]);
			//cutilSafeCall(cudaFree(d_signal[0]));
			//cutilSafeCall(cudaFree(d_signal[1]));
			cudaFree(d_signal[0]);
			cudaFree(d_signal[1]);
		}

		float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);
		for (int i = 0; i < new_size; i++) {
			outputsx[i]=(tresultsx[i]/maxot);
			outputdx[i]=(tresultdx[i]/maxot);
		}

		//printf("Freeing resources.\n");
		//Destroy CUFFT context
		//cufftSafeCall(cufftDestroy(plan));
		cufftDestroy(plan);
		// cleanup memory
		free(h_filter_kernels[0]);
		free(h_filter_kernels[1]);
		//cutilSafeCall(cudaFree(d_filter_kernels[0]));
		//cutilSafeCall(cudaFree(d_filter_kernels[1]));
		cudaFree(d_filter_kernels[0]);
		cudaFree(d_filter_kernels[1]);
		free(tresultsx);
		free(tresultdx);
		cudaDeviceSynchronize();
		//cutilDeviceReset(); 
		cudaDeviceReset(); 


		return new_size;
	}

	return -1.0f;
}


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;  
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

// Complex pointwise multiplication
// Based on ComplexPointwiseMulAndScale but without scaling... It creates more problems than it solves...
static __global__ void ComplexPointwiseMul(cufftComplex* a, const cufftComplex* b, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
		a[i] =ComplexMul(a[i], b[i]); 
} 