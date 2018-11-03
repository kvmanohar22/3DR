cmake_minimum_required(VERSION 3.0)

project(3DReconstruction)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

set(HEADERS "include")
include_directories(${HEADERS} ${OpenCV_INCLUDE_DIRS})

set(SRCS src/stitch.cpp)
add_library(RECONSTRUCT_LIBS SHARED ${SRCS})

list(APPEND LIBS ${OpenCV_LIBRARIES} RECONSTRUCT_LIBS)
add_subdirectory(tests)