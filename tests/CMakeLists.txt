function(tester target src)
   add_executable(${target} ${src})
   target_link_libraries(${target} ${LIBS})
endfunction()

# Geometric transformations
tester(test_scale test_scale.cpp)
tester(test_rotate test_rotate.cpp)
tester(test_translate test_translate.cpp)

# Some math and utils
tester(test_math test_math.cpp)
tester(test_svd test_SVD.cpp)
tester(test_utils test_utils.cpp)
tester(test_opencv test_opencv.cpp)
tester(test_viewer test_viewer.cpp)

# Image stitching and Panorama
tester(test_stitch test_stitch.cpp)
tester(test_panorama test_panorama.cpp)

# TwoView SfM
tester(test_two test_twoview.cpp)
