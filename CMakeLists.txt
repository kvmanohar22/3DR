cmake_minimum_required(VERSION 3.0)

project(3DReconstruction)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

set(HEADERS "include")
include_directories(${HEADERS} ${OpenCV_INCLUDE_DIRS})

set(LIBS ${OpenCV_LIBRARIES})
add_subdirectory(tests)