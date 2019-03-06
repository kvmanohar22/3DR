cmake_minimum_required(VERSION 3.0)

project(3DR)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR /usr/local/share/OpenCV)
set(Pangolin_DIR /usr/local/share/Pangolin/)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Pangolin REQUIRED)

set(HEADERS "include")
set(GLAD_SRC src/glad.c)
set(LIBS ${OpenCV_LIBRARIES} glfw)
include_directories(${HEADERS} ${OpenCV_INCLUDE_DIRS} ${GLFW3_INCLUDE_DIRS})

set(SRCS1 src/panorama.cpp src/utils.cpp)
add_library(RECONSTRUCT_LIBS1 SHARED ${SRCS1})

set(UTILS src/utils.cpp)
add_library(RECONSTRUCT_UTILS SHARED ${UTILS})

set(SRCS2 src/stitch.cpp)
add_library(RECONSTRUCT_LIBS2 SHARED ${SRCS2})

list(APPEND LIBS ${OpenCV_LIBRARIES} RECONSTRUCT_LIBS1 RECONSTRUCT_LIBS2 RECONSTRUCT_UTILS)
add_subdirectory(tests)
