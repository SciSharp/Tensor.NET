
file(GLOB_RECURSE SOURCES common/*.cpp naive/*.cpp)

enable_testing()

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_executable( tensornet_core_test ${SOURCES} )
    target_link_libraries(
      tensornet_core_test
      PRIVATE tensornet_core_base
      PRIVATE tensornet_core_op
      PRIVATE gtest_main
    )
    set_target_properties(tensornet_core_test PROPERTIES
        SUFFIX "exe"
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(tensornet_core_test PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_executable(tensornet_core_test ${SOURCES})
    target_link_libraries(
      tensornet_core_test
      PRIVATE tensornet_core_base
      PRIVATE tensornet_core_op
      PRIVATE gtest_main
    )
    target_include_directories(tensornet_core_test PRIVATE ${PROJECT_SOURCE_DIR})
endif()

include(GoogleTest)
gtest_discover_tests(tensornet_core_test)
  