// GPU/CPU Convolution engine
//  OCLconv.cpp
//  OCLconv
//
//  Created by Davide Andrea Mauro on 2011-07-29.
//	Last Edited by Davide Andrea Mauro on 2013-03-05.
//

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdarg.h>



// includes, project
// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

#include <clFFT.h>
#include <OCLconv.hpp>
// standard utilities and systems includes
//#include <oclUtils.h>

typedef enum {
    clFFT_OUT_OF_PLACE,
    clFFT_IN_PLACE,
}clFFT_TestType; 

typedef struct
{
    double real;
    double imag;
}clFFT_ComplexDouble; 

typedef struct
{
    double *real;
    double *imag;
}clFFT_SplitComplexDouble; 


cl_device_type getGlobalDeviceType()
{
    char *force_cpu = getenv( "CL_DEVICE_TYPE" );
    if( force_cpu != NULL )
    {
        if( strcmp( force_cpu, "gpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_GPU" ) == 0 )
            return CL_DEVICE_TYPE_GPU;
        else if( strcmp( force_cpu, "cpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_CPU" ) == 0 )
            return CL_DEVICE_TYPE_CPU;
        else if( strcmp( force_cpu, "accelerator" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            return CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( force_cpu, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
          return CL_DEVICE_TYPE_DEFAULT;
    }
    // default
    return CL_DEVICE_TYPE_GPU;
}

int isPowerOfTwo (int x){
int powerOfTwo = 1;

 while (powerOfTwo < x && powerOfTwo)
   powerOfTwo *= 2;
 return (powerOfTwo);
}

    //////////////////////////////////////////////////////////////////////////////
    //! Loads a Program file and prepends the cPreamble to the code.
    //!
    //! @return the source string if succeeded, 0 otherwise
    //! @param cFilename        program filename
    //! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
    //! @param szFinalLength    returned length of the code string
    //////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
        // locals
    FILE* pFileStream = NULL;
    size_t szSourceLength;
    
        // open the OpenCL source code file
#ifdef _WIN32   // Windows version
    if(fopen_s(&pFileStream, cFilename, "rb") != 0)
    {
        return NULL;
    }
#else           // Linux version
    pFileStream = fopen(cFilename, "rb");
    if(pFileStream == 0)
    {
        return NULL;
    }
#endif
    
    size_t szPreambleLength = strlen(cPreamble);
    
        // get the length of the source code
    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);
    
        // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }
    
        // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';
    
    return cSourceString;
}


    // Round Up Division function
size_t shrRoundUp(int group_size, int global_size)
{
    int r = global_size % group_size;
    if(r == 0)
    {
        return global_size;
    } else
    {
        return global_size + group_size - r;
    }
}


    //////////////////////////////////////////////////////////////////////////////
    //! Find the path for a file assuming that
    //! files are found in the searchPath.
    //!
    //! @return the path if succeeded, otherwise 0
    //! @param filename         name of the file
    //! @param executable_path  optional absolute path of the executable
    //////////////////////////////////////////////////////////////////////////////
char* shrFindFilePath(const char* filename, const char* executable_path)
{
        // <executable_name> defines a variable that is replaced with the name of the executable
    using namespace std;
        // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
        // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char* searchPath[] =
    {
        "./",                                       // same dir
        "./data/",                                  // "/data/" subdir
        "./src/",                                   // "/src/" subdir
        "./src/<executable_name>/data/",            // "/src/<executable_name>/data/" subdir
        "./inc/",                                   // "/inc/" subdir
        "../",                                      // up 1 in tree
        "../data/",                                 // up 1 in tree, "/data/" subdir
        "../src/",                                  // up 1 in tree, "/src/" subdir
        "../inc/",                                  // up 1 in tree, "/inc/" subdir
        "../OpenCL/src/<executable_name>/",         // up 1 in tree, "/OpenCL/src/<executable_name>/" subdir
        "../OpenCL/src/<executable_name>/data/",    // up 1 in tree, "/OpenCL/src/<executable_name>/data/" subdir
        "../OpenCL/src/<executable_name>/src/",     // up 1 in tree, "/OpenCL/src/<executable_name>/src/" subdir
        "../OpenCL/src/<executable_name>/inc/",     // up 1 in tree, "/OpenCL/src/<executable_name>/inc/" subdir
        "../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir
        "../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir
        "../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir
        "../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir
        "../DirectCompute/src/<executable_name>/",      // up 1 in tree, "/DirectCompute/src/<executable_name>/" subdir
        "../DirectCompute/src/<executable_name>/data/", // up 1 in tree, "/DirectCompute/src/<executable_name>/data/" subdir
        "../DirectCompute/src/<executable_name>/src/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/src/" subdir
        "../DirectCompute/src/<executable_name>/inc/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/inc/" subdir
        "../../",                                   // up 2 in tree
        "../../data/",                              // up 2 in tree, "/data/" subdir
        "../../src/",                               // up 2 in tree, "/src/" subdir
        "../../inc/",                               // up 2 in tree, "/inc/" subdir
        "../../../",                                // up 3 in tree
        "../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir
        "../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir
        "../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir
        "../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir
        "../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
        "../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
        "../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
        "../../../sandbox/<executable_name>/inc/"   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
    };
    
        // Extract the executable name
    string executable_name;
    if (executable_path != 0)
    {
        executable_name = string(executable_path);
        
#ifdef _WIN32
            // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);
        
		if (executable_name.rfind(".exe") != string::npos)
        {
                // we strip .exe, only if the .exe is found
			executable_name.resize(executable_name.size() - 4);
		}
#else
            // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0,delimiter_pos+1);
#endif
        
    }
    
        // Loop over all search paths and return the first hit
    for( unsigned int i = 0; i < sizeof(searchPath)/sizeof(char*); ++i )
    {
        string path(searchPath[i]);
        size_t executable_name_pos = path.find("<executable_name>");
        
            // If there is executable_name variable in the searchPath
            // replace it with the value
        if(executable_name_pos != string::npos)
        {
            if(executable_path != 0)
            {
                path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);
                
            }
            else
            {
                    // Skip this path entry if no executable argument is given
                continue;
            }
        }
        
            // Test if the file exists
        path.append(filename);
        fstream fh(path.c_str(), fstream::in);
        if (fh.good())
        {
                // File found
                // returning an allocated array here for backwards compatibility reasons
            char* file_path = (char*) malloc(path.length() + 1);
#ifdef _WIN32
            strcpy_s(file_path, path.length() + 1, path.c_str());
#else
            strcpy(file_path, path.c_str());
#endif
            return file_path;
        }
    }    
    
        // File not found
    return 0;
}

    // Name of the file with the source code for the computation kernel
    // *********************************************************************
const char* cSourceFile = "OCLComplexPwiseMul.cl";
cl_device_id     device_id;
cl_context       context;
cl_command_queue queue;
    //typedef unsigned long long ulong;



float OCLconv(float* input, int SIGNAL_SIZE, float* filtersx, float* filterdx, int FILTER_KERNEL_SIZE, float* outputsx, float* outputdx, int direct, char* argv) 
{

    cl_ulong gMemSize;
    clFFT_Dim3 n = { 1, 1, 1 };
    int batchSize = 1;
    cl_device_id device_ids[16];
    clFFT_Plan plan = NULL;         
           
    //Start Setup       
    cl_int err;
    unsigned int num_devices;
    cl_device_type device_type = getGlobalDeviceType();
    if(device_type != CL_DEVICE_TYPE_GPU) 
    {
        printf("Test only supported on DEVICE_TYPE_GPU\n");
        return -1;
    }
    cl_platform_id platform_ids[16];
    err= clGetPlatformIDs(sizeof(platform_ids), platform_ids, NULL);
    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, sizeof(device_ids), device_ids, &num_devices);
        //printf("%d", err);
    if(err) 
    {       
        printf("clGetComputeDevice failed\n");
        return -1;
    }
    device_id = NULL;
    
    unsigned int i;
    for(i = 0; i < num_devices; i++)
    {
        cl_bool available;
        err = clGetDeviceInfo(device_ids[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
        if(err)
        {
             printf("Cannot check device availability of device # %d\n", i);
        }
        
        if(available)
        {
            device_id = device_ids[i];
            break;
        }
        else
        {
            char name[200];
            err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
  
		 if(err == CL_SUCCESS)
            {
                 printf("Device %s not available for compute\n", name);
            }
            else
            {
                 printf("Device # %d not available for compute\n", i);
            }
        }
    }
    
    if(!device_id)
    {
        printf("None of the devices available for compute ... aborting\n");
        return -1;
    }

    

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(!context || err) 
    {
        printf("clCreateContext failed\n");
      return -1;
    }
    
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(!queue || err)
    {
        printf("clCreateCommandQueue() failed.\n");
        clReleaseContext(context);
        return -1;
    }  

    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &gMemSize, NULL);
    if(err)
    {
        printf("Failed to get global mem size\n");
        clReleaseContext(context);
        clReleaseCommandQueue(queue);
        return -1;

    }

    gMemSize /= (1024*1024);
    //End Setup
    
    // Read the OpenCL kernel in from source file
    //shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    char* cPathAndName = NULL;
    char* cSourceCL = NULL;         // Buffer to hold source for compilation
	const char* cExecutableName = NULL;
	size_t szKernelLength;			// Byte size of kernel code
	cl_program cpProgram;           // OpenCL program
	cl_kernel ckKernel;             // OpenCL kernel
	size_t szGlobalWorkSize;        // 1D var for Total # of work items
	size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	

    cPathAndName = shrFindFilePath(cSourceFile, argv);
    //oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    //oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // Create the program
    //shrLog("clCreateProgramWithSource...\n"); 
    cpProgram = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &szKernelLength, &err);

        // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    //shrLog("clBuildProgram...\n"); 
    err = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
            //shrLogEx(LOGBOTH | ERRORMSG, err, STDERROR);
            //oclLogBuildInfo(cpProgram, oclGetFirstDev(context));
            //oclLogPtx(cpProgram, oclGetFirstDev(context), "OCLconv.ptx");
        //Cleanup(EXIT_FAILURE); 
    }

    // Create the kernel
    //shrLog("clCreateKernel (ComplexPointwiseMul)...\n"); 
    ckKernel = clCreateKernel(cpProgram, "ComplexPointwiseMul", &err);

    
	if(direct==1){
	printf("Entering Direct Mode.\n"); 
	
    int proposed_length=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
    int pow2_length;
    pow2_length=isPowerOfTwo(proposed_length);
    //printf("%d , %d.\n",proposed_length, agreed_length);
    n.x=pow2_length;
	int length = n.x * n.y * n.z * batchSize;
  #ifdef _WIN32 || _WIN64
            //Host Memory
        clFFT_SplitComplex data_input_split;
        clFFT_SplitComplex data_filsx_split;
        clFFT_SplitComplex data_fildx_split;
            //clFFT_SplitComplex data_output_split = (clFFT_SplitComplex) { NULL, NULL };
        clFFT_SplitComplex data_outputsx_split;
        clFFT_SplitComplex data_outputdx_split;
#else
	//Host Memory
  	clFFT_SplitComplex data_input_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_filsx_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_fildx_split = (clFFT_SplitComplex) { NULL, NULL };
	//clFFT_SplitComplex data_output_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_outputsx_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_outputdx_split = (clFFT_SplitComplex) { NULL, NULL };
#endif
	//Device memory
	cl_mem data_input_real = NULL;
	cl_mem data_input_imag = NULL;
	cl_mem data_filsx_real = NULL;
	cl_mem data_filsx_imag = NULL;
	cl_mem data_fildx_real = NULL;
	cl_mem data_fildx_imag = NULL;

	//Allocate Host Memory
	data_input_split.real    = (float *) malloc(sizeof(float) * length);
	data_input_split.imag    = (float *) malloc(sizeof(float) * length);
	data_filsx_split.real    = (float *) malloc(sizeof(float) * length);
	data_filsx_split.imag    = (float *) malloc(sizeof(float) * length);
	data_fildx_split.real    = (float *) malloc(sizeof(float) * length);
	data_fildx_split.imag    = (float *) malloc(sizeof(float) * length);
	
	data_outputsx_split.real    = (float *) malloc(sizeof(float) * length);
	data_outputsx_split.imag    = (float *) malloc(sizeof(float) * length);
	data_outputdx_split.real    = (float *) malloc(sizeof(float) * length);
	data_outputdx_split.imag    = (float *) malloc(sizeof(float) * length);
	
	
	if(!data_input_split.real || !data_input_split.imag || !data_filsx_split.real || !data_filsx_split.imag|| !data_fildx_split.real || !data_fildx_split.imag || !data_outputsx_split.real || !data_outputsx_split.imag|| !data_outputdx_split.real || !data_outputdx_split.imag){
			err = -1;
			printf("Out-of-Resources\n");
			return -1.0f;
	}
		
	for(i = 0; i < length; i++) {
		if (i<SIGNAL_SIZE){
			data_input_split.real[i] = input[i];
			}
			if(i<FILTER_KERNEL_SIZE){
			data_filsx_split.real[i] = filtersx[i];
			data_fildx_split.real[i] = filterdx[i];
			}
			data_input_split.imag[i] = 0.0f;
			data_filsx_split.imag[i] = 0.0f;
			data_fildx_split.imag[i] = 0.0f;	
		}
	
	//Create PLAN
	plan = clFFT_CreatePlan(context, n, clFFT_1D, clFFT_SplitComplexFormat, &err );
	if(!plan || err) 
	{
		printf("clFFT_CreatePlan failed\n");
		return -1.0f;
	}

	//Copy Memory on Device
		data_input_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.real, &err);
	    if(!data_input_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_input_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.imag, &err);
	    if(!data_input_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
	    
	    data_filsx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_filsx_split.real, &err);
	    if(!data_filsx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_filsx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_filsx_split.imag, &err);
	    if(!data_filsx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
	    
	    	    data_fildx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_fildx_split.real, &err);
	    if(!data_fildx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_fildx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_fildx_split.imag, &err);
	    if(!data_fildx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
	    
	 //Execute Forward Plans  
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_input_real, data_input_imag, data_input_real, data_input_imag, 0, NULL, NULL);
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_filsx_real, data_filsx_imag, data_filsx_real, data_filsx_imag, 0, NULL, NULL);
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_fildx_real, data_fildx_imag, data_fildx_real, data_fildx_imag, 0, NULL, NULL);
	 
	 
	 //Complex Pointwise Multiply
	 // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, length);  // rounded up to the nearest multiple of the LocalWorkSize

	// Set the Argument values
    //shrLog("clSetKernelArg 0 - 4...\n\n"); 
    err = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&data_filsx_real);
    err |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&data_filsx_imag);
    err |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&data_input_real);
    err |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&data_input_imag);
    err |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&length);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	 
	// Launch kernel
    //shrLog("clEnqueueNDRangeKernel (DotProduct)...\n"); 
    err = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
    	// Set the Argument values
    //shrLog("clSetKernelArg 0 - 4...\n\n"); 
    err = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&data_fildx_real);
    err |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&data_fildx_imag);
    err |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&data_input_real);
    err |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&data_input_imag);
    err |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&length);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	 
	// Launch kernel
    //shrLog("clEnqueueNDRangeKernel (DotProduct)...\n"); 
    err = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	 
	 //Execute Reverse Plans
	 //err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Inverse, data_input_real, data_input_imag, data_input_real, data_input_imag, 0, NULL, NULL);
 	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Inverse, data_filsx_real, data_filsx_imag, data_filsx_real, data_filsx_imag, 0, NULL, NULL);
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Inverse, data_fildx_real, data_fildx_imag, data_fildx_real, data_fildx_imag, 0, NULL, NULL);
	 	
	//Read Memory from Device
	err |= clEnqueueReadBuffer(queue, data_filsx_real, CL_TRUE, 0, length*sizeof(float), data_outputsx_split.real, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, data_filsx_imag, CL_TRUE, 0, length*sizeof(float), data_outputsx_split.imag, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, data_fildx_real, CL_TRUE, 0, length*sizeof(float), data_outputdx_split.real, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, data_fildx_imag, CL_TRUE, 0, length*sizeof(float), data_outputdx_split.imag, 0, NULL, NULL);
	
	
	//Max, but Sequential...
	float maxo[2];
	maxo[0]=0.0f;
	maxo[1]=0.0f; 
	
	for (int i=0; i<proposed_length; i++){
	if (abs(maxo[0])<=abs(data_outputsx_split.real[i])) maxo[0]=data_outputsx_split.real[i];
	if (abs(maxo[1])<=abs(data_outputdx_split.real[i])) maxo[1]=data_outputdx_split.real[i];
	}
	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);
	
	//Writing Output
	for (int i=0; i<proposed_length; i++){
	outputsx[i]= data_outputsx_split.real[i]/maxot;
	outputdx[i]= data_outputdx_split.real[i]/maxot;
	}
	//Free Resources
	clFFT_DestroyPlan(plan);
	
	free(data_input_split.real);
	free(data_input_split.imag);
	free(data_filsx_split.real);
	free(data_filsx_split.imag);
	free(data_fildx_split.real);
	free(data_fildx_split.imag);
	free(data_outputsx_split.real);
	free(data_outputsx_split.imag);
	free(data_outputdx_split.real);
	free(data_outputdx_split.imag);
	
	clReleaseMemObject(data_input_real);
	clReleaseMemObject(data_input_imag);
	clReleaseMemObject(data_filsx_real);
	clReleaseMemObject(data_filsx_imag);
	clReleaseMemObject(data_fildx_real);
	clReleaseMemObject(data_fildx_imag);
	
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	
	return proposed_length;
}	

if(direct==0){
		printf("Entering Overlap and Save Mode.\n");

 	int proposed_length=SIGNAL_SIZE+FILTER_KERNEL_SIZE-1;
    int pow2_length;
    pow2_length=isPowerOfTwo(FILTER_KERNEL_SIZE*2);
    //printf("%d , %d.\n",proposed_length, agreed_length);
    n.x=pow2_length;
	int length = n.x * n.y * n.z * batchSize;

	//Create PLAN
	plan = clFFT_CreatePlan(context, n, clFFT_1D, clFFT_SplitComplexFormat, &err );
	
      #ifdef _WIN32 || _WIN64
    	//Host Memory
	clFFT_SplitComplex data_filsx_split;
	clFFT_SplitComplex data_fildx_split;
	clFFT_SplitComplex data_input_split;
#else
	//Host Memory  
	clFFT_SplitComplex data_filsx_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_fildx_split = (clFFT_SplitComplex) { NULL, NULL };
	clFFT_SplitComplex data_input_split = (clFFT_SplitComplex) { NULL, NULL };
#endif
	//Device memory
	cl_mem data_filsx_real = NULL;
	cl_mem data_filsx_imag = NULL;
	cl_mem data_fildx_real = NULL;
	cl_mem data_fildx_imag = NULL;
	
	//Allocate Host Memory

	data_filsx_split.real    = (float *) malloc(sizeof(float) * length);
	data_filsx_split.imag    = (float *) malloc(sizeof(float) * length);
	data_fildx_split.real    = (float *) malloc(sizeof(float) * length);
	data_fildx_split.imag    = (float *) malloc(sizeof(float) * length);
	
	//Initialize Filters
		for(i = 0; i < length; i++) {
			if(i<FILTER_KERNEL_SIZE){
			data_filsx_split.real[i] = filtersx[i];
			data_fildx_split.real[i] = filterdx[i];
			}
			else{
			data_filsx_split.real[i] = 0.0f;
			data_fildx_split.real[i] = 0.0f;
			}
			data_filsx_split.imag[i] = 0.0f;
			data_fildx_split.imag[i] = 0.0f;	
		}
		
		//Copy Memory on Device
		    data_filsx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_filsx_split.real, &err);
	    if(!data_filsx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_filsx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_filsx_split.imag, &err);
	    if(!data_filsx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
	    
	    	    data_fildx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_fildx_split.real, &err);
	    if(!data_fildx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_fildx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_fildx_split.imag, &err);
	    if(!data_fildx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }	
	    
	 //Execute Forward Plans   
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_filsx_real, data_filsx_imag, data_filsx_real, data_filsx_imag, 0, NULL, NULL);
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_fildx_real, data_fildx_imag, data_fildx_real, data_fildx_imag, 0, NULL, NULL);
    
	    float maxo[2];
		maxo[0]=0.0f;
		maxo[1]=0.0f; 

		float* tresultsx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+(length)));
		float* tresultdx=(float*)malloc(sizeof(float) * (SIGNAL_SIZE+(length)));

		for(int j=0; j<(SIGNAL_SIZE+(length)); j++){ 
			tresultsx[j]=0.0f;
			tresultdx[j]=0.0f;
		}
		
				 // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, length);  // rounded up to the nearest multiple of the LocalWorkSize

	//BIG LOOP!!!
	for(int k=0; k<SIGNAL_SIZE; (k+=(length-FILTER_KERNEL_SIZE))){
		//Declare
        
              #ifdef _WIN32 || _WIN64
        
		clFFT_SplitComplex data_input_split;
        clFFT_SplitComplex data_outputsx_split;
		clFFT_SplitComplex data_outputdx_split;
#else
        
		clFFT_SplitComplex data_input_split = (clFFT_SplitComplex) { NULL, NULL };
        clFFT_SplitComplex data_outputsx_split = (clFFT_SplitComplex) { NULL, NULL };
		clFFT_SplitComplex data_outputdx_split = (clFFT_SplitComplex) { NULL, NULL };
#endif
		data_input_split.real    = (float *) malloc(sizeof(float) * length);
		data_input_split.imag    = (float *) malloc(sizeof(float) * length);
		cl_mem data_inputsx_real = NULL;
		cl_mem data_inputsx_imag = NULL;
		cl_mem data_inputdx_real = NULL;
		cl_mem data_inputdx_imag = NULL;
		data_outputsx_split.real    = (float *) malloc(sizeof(float) * length);
		data_outputsx_split.imag    = (float *) malloc(sizeof(float) * length);
		data_outputdx_split.real    = (float *) malloc(sizeof(float) * length);
		data_outputdx_split.imag    = (float *) malloc(sizeof(float) * length);
		
		//PAD!
		for(i = 0; i < length; i++) {
		if (((k+i)<SIGNAL_SIZE)&&(i<(length-FILTER_KERNEL_SIZE))){
			data_input_split.real[i] = input[k+i];
			}
			else{
			data_input_split.real[i] = 0.0f;
			}
			data_input_split.imag[i] = 0.0f;
	
		}

		//Copy Memory on Device
		data_inputsx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.real, &err);
	    if(!data_inputsx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_inputsx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.imag, &err);
	    if(!data_inputsx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
	    		data_inputdx_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.real, &err);
	    if(!data_inputdx_real || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }		
		data_inputdx_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_input_split.imag, &err);
	    if(!data_inputdx_imag || err) 
	    {
			printf("clCreateBuffer failed\n");
			return -1.0f;
	    }
		//Execute Forward Plans  
	 	err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_inputsx_real, data_inputsx_imag, data_inputsx_real, data_inputsx_imag, 0, NULL, NULL);
		err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Forward, data_inputdx_real, data_inputdx_imag, data_inputdx_real, data_inputdx_imag, 0, NULL, NULL);
		
		//ComplexPointWiseMul

	// Set the Argument values
    //shrLog("clSetKernelArg 0 - 4...\n\n"); 
    err = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&data_inputsx_real);
    err |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&data_inputsx_imag);
    err |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&data_filsx_real);
    err |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&data_filsx_imag);
    err |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&length);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	 
	// Launch kernel
    //shrLog("clEnqueueNDRangeKernel (DotProduct)...\n"); 
    err = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
    	// Set the Argument values
    //shrLog("clSetKernelArg 0 - 4...\n\n"); 
    err = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&data_inputdx_real);
    err |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&data_inputdx_imag);
    err |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&data_fildx_real);
    err |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&data_fildx_imag);
    err |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&length);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	 
	// Launch kernel
    //shrLog("clEnqueueNDRangeKernel (DotProduct)...\n"); 
    err = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    //oclCheckErrorEX(err, CL_SUCCESS, pCleanup);
	
		//Execute Reverse Plans
 	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Inverse, data_inputsx_real, data_inputsx_imag, data_inputsx_real, data_inputsx_imag, 0, NULL, NULL);
	 err |= clFFT_ExecutePlannar(queue, plan, batchSize, clFFT_Inverse, data_inputdx_real, data_inputdx_imag, data_inputdx_real, data_inputdx_imag, 0, NULL, NULL);
	
		//Read Memory from Device
	err |= clEnqueueReadBuffer(queue, data_inputsx_real, CL_TRUE, 0, length*sizeof(float), data_outputsx_split.real, 0, NULL, NULL);
	//err |= clEnqueueReadBuffer(queue, data_inputsx_imag, CL_TRUE, 0, length*sizeof(float), data_outputsx_split.imag, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, data_inputdx_real, CL_TRUE, 0, length*sizeof(float), data_outputdx_split.real, 0, NULL, NULL);
	//err |= clEnqueueReadBuffer(queue, data_inputdx_imag, CL_TRUE, 0, length*sizeof(float), data_outputdx_split.imag, 0, NULL, NULL);
	
	
				for (int i=0; i< length; i++){
				tresultsx[k+i]+=data_outputsx_split.real[i];
				tresultdx[k+i]+=data_outputdx_split.real[i];
				if (abs(maxo[0])<=abs(tresultsx[k+i])) maxo[0]=tresultsx[k+i];
				if (abs(maxo[1])<=abs(tresultdx[k+i])) maxo[1]=tresultdx[k+i];
			}
	
	//Free Local Resources
	clReleaseMemObject(data_inputsx_real);
	clReleaseMemObject(data_inputsx_imag);
	clReleaseMemObject(data_inputdx_real);
	clReleaseMemObject(data_inputdx_imag);
	free(data_input_split.real);
	free(data_input_split.imag);
	}
	
		float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);
		for (int i = 0; i < proposed_length; i++) {
			outputsx[i]=(tresultsx[i]/maxot);
			outputdx[i]=(tresultdx[i]/maxot);
		}
		//Free Resources
	clFFT_DestroyPlan(plan);
	

	free(data_filsx_split.real);
	free(data_filsx_split.imag);
	free(data_fildx_split.real);
	free(data_fildx_split.imag);

	clReleaseMemObject(data_filsx_real);
	clReleaseMemObject(data_filsx_imag);
	clReleaseMemObject(data_fildx_real);
	clReleaseMemObject(data_fildx_imag);
	
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	
	return proposed_length;
}		
	return -1.0f;
}