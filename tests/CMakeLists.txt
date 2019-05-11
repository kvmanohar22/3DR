# Geometric transformations
tester(test_scale test_scale.cpp)
tester(test_rotate test_rotate.cpp)
tester(test_translate test_translate.cpp)

# Some math and utils
tester(test_math test_math.cpp)
tester(test_svd test_SVD.cpp)
tester(test_utils test_utils.cpp)
tester(test_viewer test_viewer.cpp)
tester(test_pangolin test_pangolin.cc)

# Practice libraries
tester(test_opencv test_opencv.cpp)

# Image stitching and Panorama
tester(test_stitch test_stitch.cpp)
tester(test_panorama test_panorama.cpp)

# TwoView SfM
tester(test_two test_twoview.cpp)

# Misc
tester(test_cpp test_cpp.cc)

# fast corner detector
tester(test_fast_corners test_fast_corners.cpp)

# initial map estimation
tester(test_init_map test_init.cpp)

# Eigen
tester(test_eigen test_eigen.cpp)

add_subdirectory(ceres)
add_subdirectory(slam)
