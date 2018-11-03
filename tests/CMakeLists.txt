add_executable(scale test_scale.cpp)
add_executable(rotate test_rotate.cpp)
add_executable(translate test_translate.cpp)

target_link_libraries(scale ${LIBS})
target_link_libraries(rotate ${LIBS})
target_link_libraries(translate ${LIBS})
