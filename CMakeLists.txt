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

set(SRCS
    src/two.cpp
    src/utils.cpp
    src/stitch.cpp
    src/panorama.cpp
    )
add_library(DR3_LIBS SHARED ${SRCS})

list(APPEND LIBS
     ${OpenCV_LIBRARIES}
     DR3_LIBS
     )

add_subdirectory(tests)
