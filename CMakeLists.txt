cmake_minimum_required (VERSION 2.8)

SET ( MODULE_NAME cgpuconv )
project( ${MODULE_NAME} )

IF(UNIX)
  set ( CMAKE_CXX_FLAGS "-Wno-write-strings")
ENDIF(UNIX)

include_directories(  ${CMAKE_CURRENT_SOURCE_DIR} )
#include_directories(  ${CMAKE_CURRENT_BINARY_DIR} )

FILE (GLOB sources
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    )

FIND_PACKAGE(Boost REQUIRED)
IF (Boost_FOUND)
	INCLUDE_DIRECTORIES($ENV{BOOSTDIR}/include)
	LINK_DIRECTORIES($ENV{BOOSTDIR}/lib64)
ENDIF()

#Support for CUDA Integration
FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(GPUconv GPUconv.cu GPUconv.cuh)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

include_directories("$(PORT_AUDIO_INC)")
include_directories("$(BOOST_INCLUDEDIR)")
include_directories("$(LIBSNDFILE_INC)")
 
link_directories("$(BOOST_ROOT)/stage/lib") 	
link_directories("$(PORT_AUDIO_LIB)")
link_directories("$(LIBSNDFILE_LIB)")

add_executable(cgpuconv cgpuconv.cpp CPUconv.cpp OCLconv.cpp fft_execute.cpp fft_kernelstring.cpp fft_setup.cpp)

target_link_libraries(cgpuconv portaudio_x64 libsndfile-1)