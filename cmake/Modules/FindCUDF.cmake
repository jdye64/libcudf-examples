# - Try to find LibCUDF
# Once done this will define
#  CUDF_FOUND - System has CUDF
#  CUDF_INCLUDE_DIRS - The CUDF include directories
#  CUDF_LIBRARIES - The libraries needed to use CUDF

find_path(CUDF_INCLUDE_DIR cudf/cudf.h
          HINTS ${CONDA_INCLUDE_DIRS} /usr/local/include
          PATH_SUFFIXES cudf)

find_library(CUDF_LIBRARY NAMES cudf libcudf
             HINTS ${CONDA_LINK_DIRS} /usr/local/lib )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCUDF_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CUDF DEFAULT_MSG
                                  CUDF_LIBRARY CUDF_INCLUDE_DIR)

mark_as_advanced(CUDF_INCLUDE_DIR CUDF_LIBRARY )

set(CUDF_LIBRARIES ${CUDF_LIBRARY} )
set(CUDF_INCLUDE_DIRS ${CUDF_INCLUDE_DIR} )
