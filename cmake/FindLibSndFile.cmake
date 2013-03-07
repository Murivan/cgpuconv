# Base Io build system
# Written by Jeremy Tregunna <jeremy.tregunna@me.com>
#
# Find libsndfile.

#FIND_PATH(LIBSNDFILE_INCLUDE_DIR sndfile.h)

#SET(LIBSNDFILE_NAMES ${LIBSNDFILE_NAMES} sndfile libsndfile)
#FIND_LIBRARY(LIBSNDFILE_LIBRARY NAMES ${LIBSNDFILE_NAMES} PATH)

#IF(LIBSNDFILE_INCLUDE_DIR AND LIBSNDFILE_LIBRARY)
#	SET(LIBSNDFILE_FOUND TRUE)
#ENDIF(LIBSNDFILE_INCLUDE_DIR AND LIBSNDFILE_LIBRARY)

#IF(LIBSNDFILE_FOUND)
#	IF(NOT LibSndFile_FIND_QUIETLY)
#		MESSAGE(STATUS "Found LibSndFile: ${LIBSNDFILE_LIBRARY}")
#	ENDIF (NOT LibSndFile_FIND_QUIETLY)
#ELSE(LIBSNDFILE_FOUND)
#	IF(LibSndFile_FIND_REQUIRED)
#		MESSAGE(FATAL_ERROR "Could not find sndfile")
#	ENDIF(LibSndFile_FIND_REQUIRED)
#ENDIF (LIBSNDFILE_FOUND)

# - Find sndfile
# Find the native sndfile includes and libraries
#
#  SNDFILE_INCLUDE_DIR - where to find sndfile.h, etc.
#  SNDFILE_LIBRARIES   - List of libraries when using libsndfile.
#  SNDFILE_FOUND       - True if libsndfile found.

if(SNDFILE_INCLUDE_DIR)
    # Already in cache, be silent
    set(SNDFILE_FIND_QUIETLY TRUE)
endif(SNDFILE_INCLUDE_DIR)

find_path(SNDFILE_INCLUDE_DIR sndfile.h)

find_library(SNDFILE_LIBRARY NAMES sndfile)

# Handle the QUIETLY and REQUIRED arguments and set SNDFILE_FOUND to TRUE if
# all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNDFILE DEFAULT_MSG
    SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)

if(SNDFILE_FOUND)
  set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
else(SNDFILE_FOUND)
  set(SNDFILE_LIBRARIES)
endif(SNDFILE_FOUND)

mark_as_advanced(SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)
