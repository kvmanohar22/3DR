cmake_minimum_required(VERSION 3.0)

project(3DReconstruction)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

set(HEADERS "include")
include_directories(${HEADERS} ${OpenCV_INCLUDE_DIRS})

set(SRCS1 src/panorama.cpp)
add_library(RECONSTRUCT_LIBS1 SHARED ${SRCS1})

set(SRCS2 src/stitch.cpp)
add_library(RECONSTRUCT_LIBS2 SHARED ${SRCS2})

list(APPEND LIBS ${OpenCV_LIBRARIES} RECONSTRUCT_LIBS1 RECONSTRUCT_LIBS2)
add_subdirectory(tests)