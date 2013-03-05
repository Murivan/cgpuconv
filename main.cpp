// GPU/CPU Convolution engine
//  main.cpp
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-03-05.
//

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

namespace po= boost::program_options;


int main (int argc, char * const argv[]){
	clock_t start;
	start=clock();
	SNDFILE      *infile1, *infile2;
	SF_INFO      sfinfo1, sfinfo2 ;
	int          samp1, samp2;
	int          SIGNAL_SIZE, FILTER_KERNEL_SIZE;
	int          chan1, chan2, sampleread;
    
	if (argc!=6)
	{
		printf("Wrong number of arguments. Must be 5. A mono file, a mono/stereo impulse response, an output destination, a mode (0 for CUDA 1 for CPU 2 for OpenCL) and 0 for Overlap and Save or 1 for Direct.\n");
		return 1;   
	}
    
	infile1 = sf_open (argv[1], SFM_READ, &sfinfo1);
	if (infile1 == NULL)
	{   /* Failed opening --> Error Message */
		printf ("Unable to open file 1.\n");
		return  1 ;
	}
	infile2 = sf_open (argv[2], SFM_READ, &sfinfo2);
	if (infile2 == NULL)
	{   /* Failed opening --> Error Message */
		printf ("Unable to open file 2.\n");
		return  1 ;
	}
    
    
	samp1=sfinfo1.samplerate;
	samp2=sfinfo2.samplerate;
	chan1=sfinfo1.channels;
	chan2=sfinfo2.channels;
	SIGNAL_SIZE= sfinfo1.frames;
	FILTER_KERNEL_SIZE= sfinfo2.frames;
    
	if (samp1!=samp2){
		printf("Error, Sample Rate mismatch.\n");
		return 1;
	}
    
	if (((chan1==2)&&(chan2==2))||((chan1>1)||(chan2>2))){
		printf("Unable to perform convolution with two stereo or multichannel files.\n");
		return 1;
	}
    
	float* input1= (float*)malloc(sizeof(float) * SIGNAL_SIZE*chan1);
	// Allocate host memory for the signal
	sampleread = sf_read_float (infile1, input1, SIGNAL_SIZE*chan1);
	if (sampleread != SIGNAL_SIZE ) 
	{ 
		printf ("Error!");
		return 1;
	}
    
	float* input2= (float*)malloc(sizeof(float) * FILTER_KERNEL_SIZE*chan2);
	// Allocate host memory for the filter  
	sampleread = sf_read_float (infile2, input2, FILTER_KERNEL_SIZE*chan2);
	if (sampleread != FILTER_KERNEL_SIZE*chan2 ) 
	{ 
		printf ("Error!");
		return 1;
	}
    
    
	float* filtersx= (float*)malloc(sizeof(float) * FILTER_KERNEL_SIZE);
	float* filterdx= (float*)malloc(sizeof(float) * FILTER_KERNEL_SIZE);
    
	for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
		if(chan2==2){
			filtersx[i] = input2[2*i];
			filterdx[i] = input2[2*i+1];
		}
		else{
			filtersx[i] = input2[i];
			filterdx[i] = input2[i];
		}
	}
	float* outputsx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+FILTER_KERNEL_SIZE-1));
	float* outputdx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+FILTER_KERNEL_SIZE-1));
    
	float new_size=-1.0f;
    clock_t intermediate_start, intermediate_end;
	int direct=0;
	if(argv[5][0]=='1'){
		direct=1;
	}
	else if(argv[5][0]!='0'){
		printf("Wrong mode.\n");
		return 1;
	}
	// Last argument 1 for direct 0 for overlap&save
	if(argv[4][0]=='0'){
		printf("Entering GPU mode.\n");
		intermediate_start= clock();
		new_size=GPUconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct); 
		intermediate_end= clock();
        
	}
	else if(argv[4][0]=='1'){
		printf("Entering CPU mode.\n");
		intermediate_start= clock();
		new_size=CPUconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct);
		intermediate_end= clock();
	}
	else if(argv[4][0]=='2'){
	intermediate_start= clock();
		printf("Entering OpenCL mode.\n");
		new_size=OCLconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct, argv[0]);
		intermediate_end= clock();
	}
	else{
		printf("Wrong mode.\n");
		return 1;
	}
	//printf("Out of Convolution.\n");
	if (new_size<=0){
		printf("Error in GPU or CPU module.\n");
		return 1;
	}
	//printf("Writing to disk.\n");
    
	SF_INFO sfinfout;
	sfinfout.channels=2;
	sfinfout.samplerate=sfinfo1.samplerate;
	sfinfout.format=sfinfo1.format;
	SNDFILE *outfile;
	outfile = sf_open (argv[3], SFM_WRITE, &sfinfout);
	float* output=(float*)malloc(sizeof(float) * new_size * sfinfout.channels);
	for (int i = 0; i < new_size; i++) {
		output[2*i]=(outputsx[i]);
		output[2*i+1]=(outputdx[i]);
	}
	sf_write_float(outfile, output, new_size*sfinfout.channels);
    
	sf_close(infile1);
	sf_close(infile2);
	sf_close(outfile);
    free(input1);
    free(input2);
    free(filtersx);
    free(filterdx);
    free(outputsx);
    free(outputdx);
    free(output);
	clock_t end;
	end = clock();
	printf("Global Time: %lf seconds, Partial: %lf seconds.\n", ((float)(end-start)/CLOCKS_PER_SEC), ((float)(intermediate_end-intermediate_start)/CLOCKS_PER_SEC));
	return 0;     
}