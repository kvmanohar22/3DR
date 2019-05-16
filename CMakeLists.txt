cmake_minimum_required(VERSION 3.0)

project(3DR)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenCV_DIR /usr/local/share/OpenCV)
set(Pangolin_DIR /usr/local/share/Pangolin/)
set(vikit_DIR /home/kv/slam/svo/rpg_vikit/vikit_common/build)

find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(fast REQUIRED)
find_package(vikit_common REQUIRED)
find_package(Sophus REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS "include")
set(LIBS ${OpenCV_LIBRARIES} glfw)
include_directories(${HEADERS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${GLFW3_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${CERES_INCLUDE_DIRS}
                    ${fast_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${vikit_common_INCLUDE_DIRS}
                    ${Eigen3_INCLUDE_DIRS})

set(SRCS
   src/two.cpp
   src/utils.cpp
   src/stitch.cpp
   src/panorama.cpp
   src/frame.cpp
   src/point.cpp
   src/map.cpp
   src/timer.cpp
   src/viewer.cpp
   src/config.cpp
   src/initialization.cpp
   src/handler.cpp
   src/features.cpp
   src/camera.cpp)
add_library(DR3_LIBS SHARED ${SRCS})

list(APPEND LIBS
   DR3_LIBS
   ${OpenCV_LIBRARIES}
   ${Pangolin_LIBRARIES}
   ${fast_LIBRARIES}
   ${vikit_common_LIBRARIES}
   ${Eigen3_LIBS}
   ${Sophus_LIBRARIES}
   ${CERES_LIBRARIES})

function(tester target src)
   add_executable(${target} ${src})
   target_link_libraries(${target} ${LIBS})
endfunction()

add_subdirectory(tests)
