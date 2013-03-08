// GPU/CPU Convolution engine
//  cgpuconv.cpp
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-03-04.
//

#include <cgpuconv.hpp>

int main (int argc, char * const argv[]){
	clock_t start;
	start=clock();
	SNDFILE      *infile1, *infile2;
	SF_INFO      sfinfo1, sfinfo2 ;
	int          samp1, samp2;
	int          SIGNAL_SIZE, FILTER_KERNEL_SIZE;
	int          chan1, chan2, sampleread;
    
    int mode, target, azimuth, elevation;
	float distance;
    string oinfile, ooutput, oimpulse;
    char *leftfir, *rightfir;


        // Declare a group of options that will be
        // allowed only on command line
    po::options_description generic("Generic options");
    generic.add_options()
    ("version,v", "print version string")
    ("help,h", "produce help message")
    ;
    
        // Declare a group of options that will be
        // allowed both on command line and in
        // config file
    po::options_description config("Configuration");
    config.add_options()
    ("input,i", po::value<string>(&oinfile)->implicit_value(" "), "Mono Input file path")
    ("impulse,p", po::value<string>(&oimpulse)->implicit_value(" "), "Mono/Stereo Impulse Response path")
    ("output,o", po::value<string>(&ooutput)->implicit_value(" "), "Output file path")
    ("mode,m", po::value<int>(&mode)->default_value(0), "0 for Overlap and Save or 1 for Direct.")
    ("target,t", po::value<int>(&target)->default_value(0), "0 for CPU, 1 for CUDA, or 2 for OpenCL.")
    ("leftfir,l", po::value<char *>(&leftfir)->default_value(".\\HRTF\\sub1_L.wav"), "Left FIR path")
    ("rightfir,r", po::value<char *>(&rightfir)->default_value(".\\HRTF\\sub1_R.wav"), "Right FIR path")
    ("azimuth,a", po::value<int>(&azimuth)->default_value(0), "Azimuth")
    ("elevation,e", po::value<int>(&elevation)->default_value(0), "Elevation")
    ("distance,d", po::value<float>(&distance)->default_value(1.0), "Distance")
    ;
    

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config);
    
    po::options_description config_file_options;
    config_file_options.add(config);

    po::options_description visible("Allowed options");
    visible.add(generic).add(config);

	po::variables_map vm;
    try
    {
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    //po::store(po::parse_config_file("cgpuconv.cfg", config_file_options, false), vm);
	po::notify(vm);
    }
    
    catch(po::required_option& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return ERROR_IN_COMMAND_LINE;
    }
    catch(po::error& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return ERROR_IN_COMMAND_LINE;
    }
    
	if (vm.count("help")) {
		cout << visible << "\n";
		return 1;
	}
	if (vm.count("version")) {
		cout << "Version 0.4 Beta by Davide Andrea Mauro." << "\n";
		return 1;
	}
	else {
		cout << "Starting with default parameters.\n";
	}

//	if (argc!=6)
//	{
//		printf("Wrong number of arguments. Must be 5. A mono file, a mono/stereo impulse response, an output destination, a mode (0 for CUDA 1 for CPU 2 for OpenCL) and 0 for Overlap and Save or 1 for Direct.\n");
//		return 1;   
//	}
    
	infile1 = sf_open (oinfile.c_str(), SFM_READ, &sfinfo1);
	if (infile1 == NULL)
	{   /* Failed opening --> Error Message */
		printf ("Unable to open file 1.\n");
		return  1 ;
	}
	infile2 = sf_open (oimpulse.c_str(), SFM_READ, &sfinfo2);
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
	if(mode==1){
		direct=1;
	}
	else if(mode!=0){
		printf("Wrong mode.\n");
		return 1;
	}
	// Last argument 1 for direct 0 for overlap&save
	if(target==0){
        printf("Entering CPU mode.\n");
		intermediate_start= clock();
		new_size=CPUconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct);
		intermediate_end= clock();

        
	}
	else if(target==1){
		printf("Entering GPU mode.\n");
		intermediate_start= clock();
		new_size=GPUconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct);
		intermediate_end= clock();
	}
	else if(target==2){
	intermediate_start= clock();
		printf("Entering OpenCL mode.\n");
		new_size=OCLconv(input1, SIGNAL_SIZE, filtersx, filterdx, FILTER_KERNEL_SIZE, outputsx, outputdx, direct, argv[0]);
		intermediate_end= clock();
	}
	else{
		printf("Wrong target.\n");
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
	outfile = sf_open (ooutput.c_str(), SFM_WRITE, &sfinfout);
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