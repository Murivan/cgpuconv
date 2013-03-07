cmake_minimum_required (VERSION 2.8)

SET ( MODULE_NAME cgpuconv )
project( ${MODULE_NAME} )

IF(UNIX)
  set ( CMAKE_CXX_FLAGS "-Wno-write-strings")
ENDIF(UNIX)

# additional CMake modules
set(CMAKE_MODULE_PATH "${cgpuconv_SOURCE_DIR}/cmake")

include_directories(  ${CMAKE_CURRENT_SOURCE_DIR} )
#include_directories(  ${CMAKE_CURRENT_BINARY_DIR} )

FILE (GLOB sources
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    )



#Support for CUDA Integration
FIND_PACKAGE(CUDA REQUIRED)
if(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(GPUconv GPUconv.cu)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()


FIND_PACKAGE(Boost REQUIRED)
IF (Boost_FOUND)
	INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR})
	LINK_DIRECTORIES( ${Boost_LIBRARY_DIR})
ENDIF()

FIND_PACKAGE(PortAudio REQUIRED)
IF (PORTAUDIO_FOUND)
	INCLUDE_DIRECTORIES(${PORTAUDIO_INCLUDE_DIR})
	LINK_DIRECTORIES(${PORTAUDIO_LIBRARY_DIR})
ENDIF()

FIND_PACKAGE(LibSndFile REQUIRED)
IF (SNDFILE_FOUND)
	INCLUDE_DIRECTORIES(${SNDFILE_INCLUDE_DIR})
	LINK_DIRECTORIES(${SNDFILE_LIBRARY})
ENDIF()



add_executable(cgpuconv cgpuconv.cpp CPUconv.cpp OCLconv.cpp fft_execute.cpp fft_kernelstring.cpp fft_setup.cpp)

target_link_libraries(cgpuconv ${Boost_LIBRARIES} ${PORTAUDIO_LIBRARIES} ${SNDFILE_LIBRARIES})