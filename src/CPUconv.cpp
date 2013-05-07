// GPU/CPU Convolution engine
//  CPUconv.cpp
//  CPUconv
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
#include <complex>
#include <fftw3.h>
#include <CPUconv.hpp>
float CPUconv(float* input, int SIGNAL_SIZE, float* filtersx, float* filterdx, int FILTER_KERNEL_SIZE, float* outputsx, float* outputdx, int direct) 
{	
	if(direct==1){
		printf("Entering Direct Mode.\n");
		int new_size=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
		int mem_size=sizeof(fftw_complex)*new_size;
		fftw_plan       planinput, planfiltersx, planfilterdx;
		fftw_plan       planresultsx, planresultdx;


		//Allocate Memory
		fftw_complex* h_signal = (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* h_filter_kernelsx=(fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* h_filter_kerneldx=(fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* h_resultsx= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* h_resultdx= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* fft_resultinput= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* fft_resultfilsx= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* fft_resultfildx= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* fft_intermfilsx= (fftw_complex*)fftw_malloc(mem_size);
		fftw_complex* fft_intermfildx= (fftw_complex*)fftw_malloc(mem_size);

		//Creating Plans FW
		planinput = fftw_plan_dft_1d(new_size, h_signal, fft_resultinput, FFTW_FORWARD, FFTW_ESTIMATE);
		//planinput = fftw_plan_dft_r2c_1d(new_size, h_signal, fft_resultinput, FFTW_ESTIMATE);
		planfiltersx = fftw_plan_dft_1d(new_size, h_filter_kernelsx, fft_resultfilsx, FFTW_FORWARD, FFTW_ESTIMATE);
		//planfiltersx = fftw_plan_dft_r2c_1d(new_size, h_filter_kernelsx, fft_resultfilsx, FFTW_ESTIMATE);
		//planfilterdx = fftw_plan_dft_r2c_1d(new_size, h_filter_kerneldx, fft_resultfildx, FFTW_ESTIMATE);
		planfilterdx = fftw_plan_dft_1d(new_size, h_filter_kerneldx, fft_resultfildx, FFTW_FORWARD, FFTW_ESTIMATE);
		//Creating Plans RV
		//planresultsx = fftw_plan_dft_c2r_1d(new_size, fft_intermfilsx, h_resultsx, FFTW_ESTIMATE);
		planresultsx = fftw_plan_dft_1d(new_size, fft_intermfilsx, h_resultsx, FFTW_BACKWARD, FFTW_ESTIMATE);
		//planresultdx = fftw_plan_dft_c2r_1d(new_size, fft_intermfildx, h_resultdx, FFTW_ESTIMATE);
		planresultdx = fftw_plan_dft_1d(new_size, fft_intermfildx, h_resultdx, FFTW_BACKWARD, FFTW_ESTIMATE);
		//planresultsx = fftw_plan_dft_1d(new_size, fft_resultfilsx, h_resultsx, FFTW_BACKWARD, FFTW_ESTIMATE);
		//planresultdx = fftw_plan_dft_1d(new_size, fft_resultfildx, h_resultdx, FFTW_BACKWARD, FFTW_ESTIMATE);

		//First thing to do: PAD!


		// Initalize the memory for the signal
		for (int i = 0; i < new_size; i++) {
			if (i<SIGNAL_SIZE){
				h_signal[i][0] = input[i];
			}
			else{
				h_signal[i][0] = 0.0f;
			}
			h_signal[i][1] = 0.0f;
		}

		// Initalize the memory for the filter
		for (int i = 0; i < new_size; i++) {
			if(i<FILTER_KERNEL_SIZE){
				h_filter_kernelsx[i][0] = filtersx[i];
				h_filter_kerneldx[i][0] = filterdx[i];
			}
			else{
				h_filter_kernelsx[i][0] = 0.0f;
				h_filter_kerneldx[i][0] = 0.0f;
			}
			h_filter_kernelsx[i][1] = 0.0f;
			h_filter_kerneldx[i][1] = 0.0f;
		}

		//Execute Plans FW
		fftw_execute(planinput);
		fftw_execute(planfiltersx);
		fftw_execute(planfilterdx);

		for (int i=0; i< new_size; i++){
			fft_intermfilsx[i][0]= ((fft_resultfilsx[i][0]*fft_resultinput[i][0])-(fft_resultfilsx[i][1]*fft_resultinput[i][1]));
			fft_intermfildx[i][0]= ((fft_resultfildx[i][0]*fft_resultinput[i][0])-(fft_resultfildx[i][1]*fft_resultinput[i][1]));
			fft_intermfilsx[i][1]= ((fft_resultfilsx[i][0]*fft_resultinput[i][1])+(fft_resultfilsx[i][1]*fft_resultinput[i][0]));
			fft_intermfildx[i][1]= ((fft_resultfildx[i][0]*fft_resultinput[i][1])+(fft_resultfildx[i][1]*fft_resultinput[i][0]));
		}	
		//Allocating memory back

		// Execute Plans RV
		fftw_execute(planresultsx);
		fftw_execute(planresultdx);

		float maxo[2];
		maxo[0]=0.0f;
		maxo[1]=0.0f;  

		for (int i = 0; i < new_size; i++){
			if (abs(maxo[0])<=abs(h_resultsx[i][0])) maxo[0]=h_resultsx[i][0];
			if (abs(maxo[1])<=abs(h_resultdx[i][0])) maxo[1]=h_resultdx[i][0];
		}
		float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

		for (int i=0; i< new_size; i++){
			float temp=0.0f;
			outputsx[i]= (float)((h_resultsx[i][0])/(maxot));
			outputdx[i]= (float)((h_resultdx[i][0])/(maxot));
		}

		//fftw_freeing memory
		fftw_destroy_plan(planinput);
		fftw_destroy_plan(planfiltersx);
		fftw_destroy_plan(planfilterdx);
		fftw_destroy_plan(planresultsx);
		fftw_destroy_plan(planresultdx);
		fftw_free(h_signal);
		fftw_free(h_filter_kernelsx);
		fftw_free(h_filter_kerneldx);
		fftw_free(fft_resultinput);
		fftw_free(fft_resultfilsx);
		fftw_free(fft_resultfildx);
		fftw_free(fft_intermfilsx);
		fftw_free(fft_intermfildx);
		fftw_free(h_resultsx);
		fftw_free(h_resultdx);

		return new_size;
	}


	if(direct==0){
		printf("Entering Overlap and Save Mode.\n");
		int output_size=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
		int mem_size=sizeof(fftw_complex)*output_size;
		int payload=FILTER_KERNEL_SIZE*2;
		int mem_pay= sizeof(fftw_complex)*payload;

		// Initalize the memory for the filter
		fftw_complex* h_filter_kernelsx=(fftw_complex*)fftw_malloc(mem_pay);
		fftw_complex* h_filter_kerneldx=(fftw_complex*)fftw_malloc(mem_pay);
		for (int i = 0; i < payload; i++) {
			if(i<FILTER_KERNEL_SIZE){
				h_filter_kernelsx[i][0] = filtersx[i];
				h_filter_kerneldx[i][0] = filterdx[i];
			}
			else{
				h_filter_kernelsx[i][0] = 0.0f;
				h_filter_kerneldx[i][0] = 0.0f;
			}
			h_filter_kernelsx[i][1] = 0.0f;
			h_filter_kerneldx[i][1] = 0.0f;
		}

		fftw_complex* fft_resultfilsx= (fftw_complex*)fftw_malloc(mem_pay);
		fftw_complex* fft_resultfildx= (fftw_complex*)fftw_malloc(mem_pay);
		fftw_plan planfiltersx, planfilterdx;
		planfiltersx = fftw_plan_dft_1d(payload, h_filter_kernelsx, fft_resultfilsx, FFTW_FORWARD, FFTW_ESTIMATE);
		planfilterdx = fftw_plan_dft_1d(payload, h_filter_kerneldx, fft_resultfildx, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(planfiltersx);
		fftw_execute(planfilterdx);

		double* tresultsx=(double*)malloc(sizeof(double) * (SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)));
		double* tresultdx=(double*)malloc(sizeof(double) * (SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)));

		for(int j=0; j<(SIGNAL_SIZE+(FILTER_KERNEL_SIZE*2)); j++){ 
			tresultsx[j]=0.0f;
			tresultdx[j]=0.0f;
		}

		float maxo[2];
		maxo[0]=0.0f;
		maxo[1]=0.0f;  

		for(int k=0; k<SIGNAL_SIZE; (k+=FILTER_KERNEL_SIZE)){
			//printf("Iteration %d.\n", k);
			fftw_plan       planinput;
			fftw_plan       planresultsx, planresultdx;
			//Allocate Memory
			fftw_complex* h_signal = (fftw_complex*)fftw_malloc(mem_pay);
			fftw_complex* h_resultsx= (fftw_complex*)fftw_malloc(mem_pay);
			fftw_complex* h_resultdx= (fftw_complex*)fftw_malloc(mem_pay);
			fftw_complex* fft_resultinput= (fftw_complex*)fftw_malloc(mem_pay);
			fftw_complex* fft_intermfilsx= (fftw_complex*)fftw_malloc(mem_pay);
			fftw_complex* fft_intermfildx= (fftw_complex*)fftw_malloc(mem_pay);

			//Creating Plans FW
			planinput = fftw_plan_dft_1d(payload, h_signal, fft_resultinput, FFTW_FORWARD, FFTW_ESTIMATE);
			//Creating Plans RV
			planresultsx = fftw_plan_dft_1d(payload, fft_intermfilsx, h_resultsx, FFTW_BACKWARD, FFTW_ESTIMATE);
			planresultdx = fftw_plan_dft_1d(payload, fft_intermfildx, h_resultdx, FFTW_BACKWARD, FFTW_ESTIMATE);


			//First thing to do: PAD!
			// Initalize the memory for the signal
			for (int i = 0; i < payload; i++) {
				if (((k+i)<SIGNAL_SIZE)&&(i<(payload/2))){
					h_signal[i][0] = input[k+i];
				}
				else{
					h_signal[i][0] = 0.0f;
				}
				h_signal[i][1] = 0.0f;
			}
			//Execute Plans FW
			fftw_execute(planinput);


			for (int i=0; i< payload; i++){
				fft_intermfilsx[i][0]= ((fft_resultfilsx[i][0]*fft_resultinput[i][0])-(fft_resultfilsx[i][1]*fft_resultinput[i][1]));
				fft_intermfildx[i][0]= ((fft_resultfildx[i][0]*fft_resultinput[i][0])-(fft_resultfildx[i][1]*fft_resultinput[i][1]));
				fft_intermfilsx[i][1]= ((fft_resultfilsx[i][0]*fft_resultinput[i][1])+(fft_resultfilsx[i][1]*fft_resultinput[i][0]));
				fft_intermfildx[i][1]= ((fft_resultfildx[i][0]*fft_resultinput[i][1])+(fft_resultfildx[i][1]*fft_resultinput[i][0]));
			}	
			//Allocating memory back
			// Execute Plans RV
			fftw_execute(planresultsx);
			fftw_execute(planresultdx);

			for (int i=0; i< payload; i++){
				tresultsx[k+i]+=h_resultsx[i][0];
				tresultdx[k+i]+=h_resultdx[i][0];
				if (abs(maxo[0])<=abs(tresultsx[k+i])) maxo[0]=tresultsx[k+i];
				if (abs(maxo[1])<=abs(tresultdx[k+i])) maxo[1]=tresultdx[k+i];
			}

			fftw_destroy_plan(planinput);
			fftw_destroy_plan(planresultsx);
			fftw_destroy_plan(planresultdx);
			fftw_free(fft_intermfilsx);
			fftw_free(fft_intermfildx);
			fftw_free(h_resultsx);
			fftw_free(h_resultdx);
			fftw_free(h_signal);
			fftw_free(fft_resultinput);

		}

		//printf("Out of Iteration.\n");


		float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);
		//printf("Out of MAX.\n");

		for (int i=0; i< output_size; i++){
			outputsx[i]= (float)((tresultsx[i])/(maxot));
			outputdx[i]= (float)((tresultdx[i])/(maxot));
		}
		//printf("Out of Copying back.\n");
		//fftw_freeing memory
		fftw_destroy_plan(planfiltersx);
		fftw_destroy_plan(planfilterdx);
		fftw_free(h_filter_kernelsx);
		fftw_free(h_filter_kerneldx);
		fftw_free(fft_resultfilsx);
		fftw_free(fft_resultfildx);
		free(tresultsx);
		free(tresultdx);

		return output_size;
	}

	return -1.0f;
}